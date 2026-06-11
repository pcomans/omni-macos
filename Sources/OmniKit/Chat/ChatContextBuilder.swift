import Foundation

/// One retrieved source passage shown to the model and cited in the answer.
public struct ChatSource: Sendable, Identifiable {
    public let id: Int          // 1-based citation number, matches "[n]" in the prompt
    public let path: String
    public let locator: String  // "Page 3" / "Line 120" / ""
    public let score: Float
    public let text: String     // full chunk text, or the stored snippet when re-derivation fell back
    public let isSnippetOnly: Bool
}

/// The assembled RAG context: the cited sources plus the user-turn string that embeds them.
public struct ChatContext: Sendable {
    public let sources: [ChatSource]
    public let userMessage: String

    /// Fixed system instruction: answer only from the provided sources, cite inline, admit when the
    /// answer is not present. Kept here so the chat model and the verifier share one definition.
    public static let systemPrompt = """
        You are a helpful assistant that answers questions about the user's local files. \
        Answer using ONLY the numbered sources below. Cite sources inline with bracketed numbers \
        like [1] or [2][3] after the statements they support. If the sources do not contain the \
        answer, say you could not find it in the indexed files. Be concise.
        """
}

/// Builds the retrieval-augmented context for a chat question.
///
/// LEARNING NOTE - retrieval-augmented generation (RAG)
/// ---------------------------------------------------
/// The chat model does not know your files. RAG bridges that: embed the question with the same model
/// that powers search, find the most similar indexed chunks, paste their text into the prompt as
/// numbered sources, and instruct the model to answer only from them and cite which it used. The
/// model's job becomes reading and synthesizing provided text, not recalling facts - which is exactly
/// what a small local model is good at, and it makes answers verifiable (every claim points at a file).
///
/// LEARNING NOTE - why we re-derive chunk text
/// -------------------------------------------
/// The index stores only a short snippet per chunk (to keep it small), not the full chunk text. So to
/// give the model real context we re-extract the original file and re-chunk it with the shared
/// ``TextChunker`` - the exact logic the indexer used - then take the chunk at the stored index. A
/// guard makes wrong-text citations impossible: we recompute the snippet of the re-derived chunk and
/// require it to equal the stored snippet. If anything drifted (the chunk-size setting changed since
/// indexing, or the file was edited), we fall back to the stored snippet and mark the source
/// `isSnippetOnly`, so the worst case is less context, never misattributed context.
///
/// LEARNING NOTE - the bibliography problem (prose re-ranking)
/// ----------------------------------------------------------
/// Pure embedding similarity has a notorious failure on books: bibliography / "further reading" /
/// index pages are packed with the exact query terms (author names, titles, "New Kingdom") and so
/// out-rank the actual narrative pages, which use varied prose. Those reference pages are useless to
/// answer from. We counter it by re-ranking the top candidates with a small penalty proportional to
/// how citation-dense a chunk looks (mainly the density of 4-digit years), pushing real prose above
/// reference lists before we fill the context budget.
public enum ChatContextBuilder {
    private struct Candidate {
        let path: String; let chunkIndex: Int; let score: Float
        let snippet: String; let locator: String; let kind: String; let modified: Double
    }

    /// Build context for `question`. `queryVector` must be the question embedded as a query
    /// (`OmniEngine.embedQuery`). Pure CPU + disk; call off the main actor.
    public static func build(question: String,
                             queryVector: [Float],
                             store: VectorStore,
                             folder: URL?,
                             settings: IndexSettings,
                             charBudget: Int = 12000,
                             maxSources: Int = 12,
                             maxFiles: Int = 8) -> ChatContext {
        var filter = SearchFilter()
        filter.folderPrefix = folder?.path
        let hits = store.search(queryVector, filter: filter, topK: 60)

        // First `maxFiles` distinct files in score order; remember each file's stored mtime + kind.
        var fileOrder: [String] = []
        var fileInfo: [String: (modified: Double, kind: String)] = [:]
        for h in hits {
            if fileInfo[h.path] == nil {
                fileInfo[h.path] = (h.modified, h.kind)
                fileOrder.append(h.path)
            }
            if fileOrder.count >= maxFiles { break }
        }

        // Gather the top chunks per file as candidates.
        var candidates: [Candidate] = []
        for path in fileOrder {
            guard let info = fileInfo[path] else { continue }
            for c in store.rankChunks(queryVector, path: path, topK: 2) {
                candidates.append(Candidate(path: path, chunkIndex: c.chunkIndex, score: c.score,
                                            snippet: c.snippet, locator: c.locator,
                                            kind: info.kind, modified: info.modified))
            }
        }
        candidates.sort { $0.score > $1.score }

        // Resolve text for the strongest candidates (capped so we don't extract endlessly), then
        // re-rank by an adjusted score that demotes citation-dense (bibliography/index) pages.
        struct Resolved { let cand: Candidate; let text: String; let snippetOnly: Bool; let adjusted: Float }
        var cache: [String: [TextChunker.Piece]?] = [:]   // per-file extraction cache (nil = unusable)
        var resolved: [Resolved] = []
        for cand in candidates.prefix(20) {
            guard let (text, snippetOnly) = resolveText(cand, settings: settings, cache: &cache) else { continue }
            resolved.append(Resolved(cand: cand, text: text, snippetOnly: snippetOnly,
                                     adjusted: cand.score - referencePenalty(text)))
        }
        resolved.sort { $0.adjusted > $1.adjusted }

        // Greedily fill the character budget in adjusted-score order; always keep at least one source.
        var sources: [ChatSource] = []
        var used = 0
        var nextId = 1
        for r in resolved {
            if sources.count >= maxSources { break }
            let header = sourceHeader(path: r.cand.path, locator: r.cand.locator)
            let cost = header.count + r.text.count
            if used + cost > charBudget && !sources.isEmpty { break }
            sources.append(ChatSource(id: nextId, path: r.cand.path, locator: r.cand.locator,
                                      score: r.cand.score, text: r.text, isSnippetOnly: r.snippetOnly))
            used += cost
            nextId += 1
        }

        return ChatContext(sources: sources, userMessage: renderUserMessage(question: question, sources: sources))
    }

    /// Penalty (subtracted from the similarity score) for chunks that look like reference lists.
    /// Driven mainly by the density of 4-digit years - bibliographies cite many, prose cites few.
    private static func referencePenalty(_ text: String) -> Float {
        let years = yearCount(text)
        let kchars = max(1.0, Double(text.count) / 1000.0)
        let perK = Double(years) / kchars
        // No penalty up to ~2 years per 1000 chars (normal prose); ramps up beyond that, capped.
        let penalty = min(0.30, max(0.0, perK - 2.0) * 0.05)
        return Float(penalty)
    }

    /// Count 4-digit years in 1500-2099 (a cheap manual scan, no regex).
    private static func yearCount(_ text: String) -> Int {
        let s = Array(text.utf8)
        var count = 0
        var i = 0
        while i < s.count {
            // a run of digits
            if s[i] >= 48 && s[i] <= 57 {
                var j = i
                while j < s.count, s[j] >= 48, s[j] <= 57 { j += 1 }
                if j - i == 4 {
                    let y = (Int(s[i]) - 48) * 1000 + (Int(s[i+1]) - 48) * 100 + (Int(s[i+2]) - 48) * 10 + (Int(s[i+3]) - 48)
                    if y >= 1500 && y <= 2099 { count += 1 }
                }
                i = j
            } else {
                i += 1
            }
        }
        return count
    }

    /// Re-derive a candidate chunk's text, or return nil to skip it. Caches extraction per file.
    private static func resolveText(_ c: Candidate, settings: IndexSettings,
                                    cache: inout [String: [TextChunker.Piece]?]) -> (text: String, snippetOnly: Bool)? {
        let url = URL(fileURLWithPath: c.path)
        let fileName = url.lastPathComponent

        // Non-text chunks (images, video, scanned PDF pages) have no extractable passage; use the
        // stored snippet if it carries information, otherwise skip (a bare filename helps no one).
        if c.kind != "text" {
            if c.snippet.isEmpty || c.snippet == fileName { return nil }
            let tag = c.kind == "image" ? "[image] " : (c.kind == "video" ? "[video] " : "[scanned] ")
            return (tag + c.snippet, true)
        }

        func snippetFallback() -> (String, Bool)? { c.snippet.isEmpty ? nil : (c.snippet, true) }

        // Cheap drift pre-check: if the file changed since indexing, do not trust the re-chunk.
        if let attrs = try? FileManager.default.attributesOfItem(atPath: c.path),
           let mtime = (attrs[.modificationDate] as? Date)?.timeIntervalSince1970,
           abs(mtime - c.modified) > 1 {
            return snippetFallback()
        }

        // Extract + chunk the file once, cached across this file's candidates.
        let pieces: [TextChunker.Piece]?
        if let cached = cache[c.path] {
            pieces = cached
        } else {
            pieces = extractPieces(url, maxChars: settings.maxCharsPerChunk)
            cache[c.path] = pieces
        }
        guard let pieces else { return snippetFallback() }

        // Guard: the re-derived chunk's snippet must equal the stored one, else something drifted.
        if c.chunkIndex < pieces.count, TextChunker.snippet(pieces[c.chunkIndex].text) == c.snippet {
            return (pieces[c.chunkIndex].text, false)
        }
        return snippetFallback()
    }

    private static func extractPieces(_ url: URL, maxChars: Int) -> [TextChunker.Piece]? {
        guard let extracted = try? FileExtractor.extract(url) else { return nil }
        switch extracted {
        case .text(let s): return TextChunker.chunk(s, maxChars: maxChars, origin: .plain)
        case .pagedText(let s, let starts): return TextChunker.chunk(s, maxChars: maxChars, origin: .paged(starts))
        default: return nil
        }
    }

    private static func sourceHeader(path: String, locator: String) -> String {
        let name = URL(fileURLWithPath: path).lastPathComponent
        return locator.isEmpty ? name : "\(name) (\(locator))"
    }

    private static func renderUserMessage(question: String, sources: [ChatSource]) -> String {
        if sources.isEmpty {
            return "Sources:\n\n(none found in the indexed files)\n\nQuestion: \(question)"
        }
        var out = "Sources:\n\n"
        for s in sources {
            out += "[\(s.id)] \(sourceHeader(path: s.path, locator: s.locator))\n\(s.text)\n\n"
        }
        out += "Question: \(question)"
        return out
    }
}
