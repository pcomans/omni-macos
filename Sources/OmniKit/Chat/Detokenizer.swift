import Foundation
import Tokenizers

/// Converts a stream of generated token ids into displayable text, one token at a time, without ever
/// emitting broken characters.
///
/// LEARNING NOTE - why naive per-token decoding breaks
/// ---------------------------------------------------
/// Qwen3 (like most modern LLMs) uses byte-level BPE: a token is a chunk of UTF-8 *bytes*, not a
/// whole character. A single character can span several tokens - an emoji or a CJK glyph is multiple
/// bytes, and the tokenizer may split those bytes across token boundaries. If you decode each token
/// id in isolation and concatenate, a multi-byte character that straddles two tokens comes out as the
/// replacement character (U+FFFD, the "?") because the first token ends mid-character.
///
/// LEARNING NOTE - the hold-back algorithm
/// ---------------------------------------
/// The fix is to decode the running list of ids and emit only the *new, complete* suffix: keep a
/// decoded prefix, append the next id, decode again, and if the new decode ends in an incomplete
/// scalar (U+FFFD) or did not grow, hold it back and wait for the next token to complete the
/// character. swift-tokenizers ships exactly this as `StreamingDetokenizer`, so we wrap it rather
/// than re-derive the byte bookkeeping; this type exists to give the rest of OmniKit a small, stable
/// surface and to host this explanation.
final class Detokenizer {
    private let inner: StreamingDetokenizer

    init(tokenizer: any Tokenizer) {
        self.inner = StreamingDetokenizer(tokenizer: tokenizer, skipSpecialTokens: true)
    }

    /// Feed one token id. Returns a displayable chunk, or nil if the buffer currently ends mid-
    /// character and the caller should feed the next token before anything is shown.
    func consume(_ id: Int) throws -> String? {
        try inner.consume(id)
    }
}
