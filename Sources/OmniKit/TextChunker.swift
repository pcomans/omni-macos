import Foundation

/// Deterministic text chunking, shared by the indexer (which embeds chunks) and the chat RAG context
/// builder (which re-derives a chunk's full text from the original file).
///
/// Both call sites MUST chunk identically: the index stores only a short snippet per chunk, so to feed
/// a retrieved chunk's full text to the chat model we re-extract the file and re-chunk it, then locate
/// the same chunk by index. Keeping the one chunking implementation here is what guarantees that
/// re-derivation lands on the exact same text the indexer embedded. The logic moved here verbatim from
/// the indexer; `Indexer.chunk` now delegates to it.
public enum TextChunker {
    /// Characters of overlap between consecutive chunks (so a passage split across a boundary is still
    /// fully present in at least one chunk).
    public static let overlap = 200
    /// Max characters of the stored per-chunk snippet (the UI preview, also used as the drift check).
    public static let snippetLength = 220

    /// What kind of position a chunk can be mapped back to, for the human-readable locator.
    public enum Origin: Sendable {
        case plain          // real text file: chunk start -> "Line N"
        case paged([Int])   // text-layer PDF: page-start character offsets -> "Page N"
        case opaque         // converted office doc: offsets map to nothing the user can see
    }

    /// One chunk plus its locator ("Line 120" / "Page 3" / "").
    public struct Piece: Sendable {
        public let text: String
        public let locator: String
        public init(text: String, locator: String) { self.text = text; self.locator = locator }
    }

    /// Split `text` into overlapping chunks of at most `maxChars` characters (floored at 200 so chunks
    /// stay meaningful). Deterministic in (text, maxChars, origin).
    public static func chunk(_ text: String, maxChars: Int, origin: Origin) -> [Piece] {
        let limit = max(200, maxChars)
        let totalCount = text.count
        if totalCount <= limit { return [Piece(text: text, locator: "")] }
        let scalars = Array(text)
        var pieces: [Piece] = []
        var start = 0
        let step = max(1, limit - overlap)
        var line = 1          // running line number at `lineMark` (plain origin; one forward pass total)
        var lineMark = 0
        func locatorFor(_ start: Int) -> String {
            switch origin {
            case .plain:
                while lineMark < start { if scalars[lineMark].isNewline { line += 1 }; lineMark += 1 }
                return "Line \(line)"
            case .paged(let starts):
                guard !starts.isEmpty else { return "" }
                var lo = 0, hi = starts.count - 1   // last page whose start offset <= chunk start
                while lo < hi { let mid = (lo + hi + 1) / 2; if starts[mid] <= start { lo = mid } else { hi = mid - 1 } }
                return "Page \(lo + 1)"
            case .opaque:
                return ""
            }
        }
        while start < scalars.count {
            let end = min(start + limit, scalars.count)
            pieces.append(Piece(text: String(scalars[start ..< end]), locator: locatorFor(start)))
            if end == scalars.count { break }
            start += step
        }
        return pieces
    }

    /// The stored snippet for a chunk: newline/tab runs collapsed to single spaces, capped to
    /// `snippetLength`. Deterministic, so re-deriving a chunk and re-computing its snippet reproduces
    /// the value stored at index time - the basis of the chat builder's drift check.
    public static func snippet(_ text: String) -> String {
        let collapsed = text.split(whereSeparator: { $0.isNewline || $0 == "\t" }).joined(separator: " ")
        return String(collapsed.prefix(snippetLength))
    }
}
