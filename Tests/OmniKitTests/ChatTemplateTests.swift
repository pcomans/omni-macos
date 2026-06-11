import XCTest
@testable import OmniKit

/// Golden-string tests for the Qwen3 chat template. No model needed: this is pure string formatting,
/// and getting it exactly right is what keeps the model in the behavior it was trained on.
final class ChatTemplateTests: XCTestCase {
    func testGenerationPromptHasEmptyThinkBlock() {
        let msgs = [
            ChatMessage(role: .system, content: "S"),
            ChatMessage(role: .user, content: "U"),
        ]
        let out = ChatTemplate.render(msgs, addGenerationPrompt: true)
        XCTAssertEqual(out,
            "<|im_start|>system\nS<|im_end|>\n" +
            "<|im_start|>user\nU<|im_end|>\n" +
            "<|im_start|>assistant\n<think>\n\n</think>\n\n")
    }

    func testTurnsOnlyWhenNoGenerationPrompt() {
        let msgs = [
            ChatMessage(role: .user, content: "hi"),
            ChatMessage(role: .assistant, content: "hello"),
            ChatMessage(role: .user, content: "again"),
        ]
        let out = ChatTemplate.render(msgs, addGenerationPrompt: false)
        XCTAssertEqual(out,
            "<|im_start|>user\nhi<|im_end|>\n" +
            "<|im_start|>assistant\nhello<|im_end|>\n" +
            "<|im_start|>user\nagain<|im_end|>\n")
    }

    func testCleanAssistantContentStripsThinkBlock() {
        XCTAssertEqual(ChatTemplate.cleanAssistantContent("<think>\n\n</think>\n\nThe answer."), "The answer.")
        XCTAssertEqual(ChatTemplate.cleanAssistantContent("<think>reasoning</think>\nThe answer."), "The answer.")
        XCTAssertEqual(ChatTemplate.cleanAssistantContent("Already clean."), "Already clean.")
    }

    func testRenderedAssistantHistoryIsCleaned() {
        // A stored assistant turn that accidentally kept a think prefix must render clean as history.
        let msgs = [ChatMessage(role: .assistant, content: "<think>\n\n</think>\n\nHi there")]
        let out = ChatTemplate.render(msgs, addGenerationPrompt: false)
        XCTAssertEqual(out, "<|im_start|>assistant\nHi there<|im_end|>\n")
    }
}
