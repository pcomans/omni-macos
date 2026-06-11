import XCTest
@testable import OmniKit

/// Model-gated: the streaming detokenizer must reconstruct exactly the same text as a one-shot decode,
/// even across multi-byte characters (CJK, emoji) that span token boundaries, and never emit the
/// Unicode replacement character. Requires the chat tokenizer; skipped when the model is absent.
final class ChatDetokenizerTests: XCTestCase {
    func testStreamingEqualsOneShotAcrossMultibyte() async throws {
        guard let dir = ChatModelLocator.resolve() else {
            throw XCTSkip("chat model not installed (set OMNI_CHAT_MODEL_DIR or download it)")
        }
        let engine = ChatEngine(modelDir: dir, gpuCoordinator: nil)
        try await engine.load()

        let samples = [
            "Hello, world! This is plain ASCII.",
            "café naïve résumé — accented Latin",
            "日本語のテキストとひらがな、カタカナ。",
            "emoji test 🎉🚀👩‍💻 and more 🇯🇵",
            "mixed 中文 with English and 🎯 symbols",
        ]
        for s in samples {
            let ids = try engine.encodeText(s)
            let oneShot = try engine.decodeTokens(ids)
            let streamed = try engine.streamDecodeTokens(ids)
            XCTAssertEqual(streamed, oneShot, "streaming decode must equal one-shot decode for: \(s)")
            XCTAssertFalse(streamed.contains("\u{FFFD}"), "streaming decode emitted U+FFFD for: \(s)")
        }
    }
}
