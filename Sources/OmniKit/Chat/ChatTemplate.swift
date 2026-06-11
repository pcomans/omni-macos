import Foundation

/// A single conversation turn handed to the chat model.
public struct ChatMessage: Sendable, Equatable {
    public enum Role: String, Sendable { case system, user, assistant }
    public var role: Role
    public var content: String
    public init(role: Role, content: String) { self.role = role; self.content = content }
}

/// Turns a list of messages into the exact token-stream-shaped string Qwen3 was trained on.
///
/// LEARNING NOTE - what a chat template is
/// ---------------------------------------
/// A base language model only continues text. To make it behave like a chat assistant, training wraps
/// each turn in special marker tokens so the model learns "this part is the system instruction, this
/// is the user, now it is my turn to answer." A chat template is just the function that serializes a
/// message list into that exact marker format. If your runtime formats turns even slightly
/// differently from training, the model's behavior degrades, so the format has to be reproduced
/// precisely. We hardcode it (rather than running the model's Jinja template) and prove it byte-for-
/// byte equal to the reference in the fixture generator.
///
/// LEARNING NOTE - Qwen3's ChatML format
/// -------------------------------------
/// Qwen3 uses "ChatML": each turn is `<|im_start|>{role}\n{content}<|im_end|>\n`. To ask the model to
/// respond you append an *open* assistant turn with no content and let it generate up to `<|im_end|>`.
///
/// LEARNING NOTE - the empty think block (forcing non-thinking mode)
/// ----------------------------------------------------------------
/// Qwen3 is a "hybrid thinking" model: by default it first emits a `<think>...</think>` reasoning
/// block and then the answer. For grounded folder Q&A we want the answer directly, no visible
/// chain-of-thought. The official non-thinking convention is to *prefill* an already-closed, empty
/// think block into the assistant turn: `<|im_start|>assistant\n<think>\n\n</think>\n\n`. The model
/// then continues straight into the answer. That exact string is our generation prompt.
public enum ChatTemplate {
    static let imStart = "<|im_start|>"
    static let imEnd = "<|im_end|>"
    /// The empty, pre-closed think block that switches Qwen3 into non-thinking mode.
    static let emptyThink = "<think>\n\n</think>\n\n"

    /// Stop tokens: `<|im_end|>` (151645) ends the turn; `<|endoftext|>` (151643) is a safety net.
    public static let stopTokenIds: [Int] = [151645, 151643]

    /// Render `messages` to the ChatML string. When `addGenerationPrompt` is true, append the open
    /// assistant turn with the empty think block so the model answers directly.
    public static func render(_ messages: [ChatMessage], addGenerationPrompt: Bool = true) -> String {
        var out = ""
        for m in messages {
            let content = m.role == .assistant ? cleanAssistantContent(m.content) : m.content
            out += "\(imStart)\(m.role.rawValue)\n\(content)\(imEnd)\n"
        }
        if addGenerationPrompt {
            out += "\(imStart)assistant\n\(emptyThink)"
        }
        return out
    }

    /// Strip a leading empty think block from a stored assistant message before re-rendering it as
    /// history. Completed assistant turns in non-thinking mode carry no think block; this guards
    /// against accidentally embedding one if a caller stored the raw generation prefix.
    public static func cleanAssistantContent(_ s: String) -> String {
        if s.hasPrefix(emptyThink) { return String(s.dropFirst(emptyThink.count)) }
        // Also handle a think block with arbitrary inner whitespace at the very start.
        if s.hasPrefix("<think>"), let r = s.range(of: "</think>") {
            let after = s[r.upperBound...].drop(while: { $0 == "\n" })
            return String(after)
        }
        return s
    }
}
