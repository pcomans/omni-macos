import XCTest
@testable import OmniKit

/// Model-gated: validates the KV cache by checking that incremental decode (prefill one token, then
/// step the rest through the cache) yields the same last-position logits as a single full-context
/// prefill. This proves the decode path with no Python in the loop. Requires the chat model; skipped
/// when it is not installed (resolve() checks OMNI_CHAT_MODEL_DIR, the app install dir, and the HF cache).
final class ChatDecoderTests: XCTestCase {
    func testKVCacheMatchesFullPrefill() async throws {
        guard let dir = ChatModelLocator.resolve() else {
            throw XCTSkip("chat model not installed (set OMNI_CHAT_MODEL_DIR or download it)")
        }
        let engine = ChatEngine(modelDir: dir, gpuCoordinator: nil)
        try await engine.load()

        // Arbitrary valid token ids (all < vocab); the decoder works on any ids.
        let ids = [785, 2390, 21428, 374, 5470, 220, 16, 19, 1207, 13]
        let full = try engine.prefillLogits(tokenIds: ids)
        let incremental = try engine.incrementalPrefillLogits(tokenIds: ids)

        XCTAssertEqual(full.count, incremental.count)
        // Batched prefill and step-by-step decode reduce in different shapes, so bf16 rounding leaves
        // a tiny gap (~5e-4) even when the cache is correct. 0.999 still catches a structural cache
        // bug (which tanks cosine), and the argmax must agree exactly (the bf16-robust signal).
        let cos = cosineSim(full, incremental)
        XCTAssertGreaterThanOrEqual(cos, 0.999, "incremental decode must match full prefill (KV cache)")
        XCTAssertEqual(argmaxIdx(full), argmaxIdx(incremental), "cache must predict the same top token")
    }
}

func cosineSim(_ a: [Float], _ b: [Float]) -> Float {
    var dot: Float = 0, na: Float = 0, nb: Float = 0
    for i in 0 ..< min(a.count, b.count) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i] }
    return dot / (na.squareRoot() * nb.squareRoot() + 1e-12)
}

func argmaxIdx(_ a: [Float]) -> Int {
    var best = 0, bestV = a[0]
    for i in 1 ..< a.count where a[i] > bestV { bestV = a[i]; best = i }
    return best
}
