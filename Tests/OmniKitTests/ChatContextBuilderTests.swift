import XCTest
@testable import OmniKit

/// Tests for the RAG context builder and the shared chunker. No chat model needed: search uses
/// hand-built vectors, and re-derivation reads a real temp file.
final class ChatContextBuilderTests: XCTestCase {
    /// `Indexer.chunk` must produce exactly what `TextChunker.chunk` does, since the chat builder
    /// relies on that equivalence to re-derive indexed text.
    func testIndexerDelegatesToTextChunker() throws {
        let store = try VectorStore(dbURL: FileManager.default.temporaryDirectory
            .appendingPathComponent("tc-\(UUID().uuidString).sqlite"))
        defer { store.close() }
        let indexer = Indexer(store: store, embedder: NullChunkEmbedder())
        var settings = IndexSettings.default
        settings.maxCharsPerChunk = 500
        let text = String(repeating: "Lorem ipsum dolor sit amet. ", count: 200)
        let viaIndexer = indexer.chunk(text, settings: settings, origin: .plain)
        let viaChunker = TextChunker.chunk(text, maxChars: 500, origin: .plain)
        XCTAssertEqual(viaIndexer.map { $0.text }, viaChunker.map { $0.text })
        XCTAssertEqual(viaIndexer.map { $0.locator }, viaChunker.map { $0.locator })
    }

    /// A retrieved chunk's full text is re-derived from the file; on a chunk-size change it falls back
    /// to the stored snippet rather than returning the wrong passage.
    func testReDerivationAndDriftFallback() throws {
        let maxChars = 400
        let body = (0 ..< 60).map { "Sentence number \($0) about project planning and budgets." }
            .joined(separator: " ")
        let fileURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("ctx-\(UUID().uuidString).txt")
        try body.write(to: fileURL, atomically: true, encoding: .utf8)
        defer { try? FileManager.default.removeItem(at: fileURL) }
        let mtime = (try FileManager.default.attributesOfItem(atPath: fileURL.path)[.modificationDate]
            as? Date)?.timeIntervalSince1970 ?? 0

        let pieces = TextChunker.chunk(body, maxChars: maxChars, origin: .plain)
        XCTAssertGreaterThan(pieces.count, 2, "need a multi-chunk file for this test")

        let store = try VectorStore(dbURL: FileManager.default.temporaryDirectory
            .appendingPathComponent("ctx-\(UUID().uuidString).sqlite"))
        defer { store.close() }
        let dim = 8
        func unit(_ i: Int) -> [Float] { var v = [Float](repeating: 0, count: dim); v[i % dim] = 1; return v }
        var chunks: [IndexedChunk] = []
        for (j, p) in pieces.enumerated() {
            chunks.append(IndexedChunk(path: fileURL.path, modified: mtime, size: body.utf8.count,
                                       kind: "text", chunkIndex: j, snippet: TextChunker.snippet(p.text),
                                       embedding: unit(j + 1), locator: p.locator))
        }
        try store.replaceMany([(fileURL.path, chunks)])

        var settings = IndexSettings.default
        settings.maxCharsPerChunk = maxChars
        // Query aligned with chunk index 1 (embedding unit(2)).
        let ctx = ChatContextBuilder.build(question: "What is the plan?", queryVector: unit(2),
                                           store: store, folder: nil, settings: settings)
        let top = try XCTUnwrap(ctx.sources.first)
        XCTAssertEqual(top.text, pieces[1].text, "top source should re-derive chunk 1's full text")
        XCTAssertFalse(top.isSnippetOnly)
        XCTAssertTrue(ctx.userMessage.contains("[1]"))
        XCTAssertTrue(ctx.userMessage.contains("Question: What is the plan?"))

        // Change the chunk size: re-derivation no longer matches the stored snippet, so it must fall
        // back to the snippet (never a wrong passage).
        settings.maxCharsPerChunk = maxChars + 137
        let drifted = ChatContextBuilder.build(question: "q", queryVector: unit(2),
                                               store: store, folder: nil, settings: settings)
        let dtop = try XCTUnwrap(drifted.sources.first)
        XCTAssertTrue(dtop.isSnippetOnly, "a chunk-size change must trigger the snippet fallback")
        XCTAssertEqual(dtop.text, TextChunker.snippet(pieces[1].text))
    }
}

/// Minimal embedder so an Indexer can be constructed for the delegation test.
private final class NullChunkEmbedder: Embedder {
    var dim: Int { 4 }
    func embedText(_ text: String, as type: OmniInputType) -> [Float] { [1, 0, 0, 0] }
    func embedTextBatch(_ texts: [String], as type: OmniInputType) -> [[Float]] { texts.map { _ in [1, 0, 0, 0] } }
    func embedTextBatches(_ batches: [[String]], as type: OmniInputType) -> [[[Float]]] {
        batches.map { embedTextBatch($0, as: type) }
    }
    func embedImage(_ image: CGImage) -> [Float]? { nil }
    func embedVideoFrames(_ frames: [CGImage]) -> [Float]? { nil }
    func embedAudio(_ url: URL) -> [Float]? { nil }
    func embedAudioMel(_ mel: [Float], frames: Int) -> [Float]? { nil }
    func embedAudioMelBatch(_ mels: [[Float]], frames: [Int]) -> [[Float]]? { nil }
}
