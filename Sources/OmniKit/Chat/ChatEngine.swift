import Foundation
import MLX
import Tokenizers

/// Sampling and length controls for one generation.
public struct ChatGenerationParams: Sendable {
    public var temperature: Float = 0.7
    public var topP: Float = 0.8
    public var maxTokens: Int = 1024
    /// nil -> a fixed default seed (generation is still varied by temperature; tests pass a seed).
    public var seed: UInt64? = nil
    public init() {}
}

/// Timing/throughput summary, delivered once when a generation finishes.
public struct ChatGenerationStats: Sendable {
    public let promptTokens: Int
    public let generatedTokens: Int
    public let prefillSeconds: Double
    public let decodeSeconds: Double
    public var decodeTokensPerSecond: Double { decodeSeconds > 0 ? Double(generatedTokens) / decodeSeconds : 0 }
}

/// Owns the chat language model and turns a conversation into a stream of text.
///
/// LEARNING NOTE - the whole pipeline, in one place
/// ------------------------------------------------
/// A chat response is produced like this:
///   1. ``ChatTemplate`` renders the messages into Qwen3's exact ChatML string.
///   2. The tokenizer turns that string into token ids.
///   3. ``Qwen3ChatDecoder/prefill(_:cache:)`` runs the whole prompt once, filling the ``KVCache``
///      and giving the logits for the first new token.
///   4. ``Sampler`` picks a token from those logits.
///   5. ``Detokenizer`` turns the chosen id into displayable text (held back across multi-byte
///      characters) and we yield it to the UI.
///   6. ``Qwen3ChatDecoder/step(_:cache:)`` runs that one token to get the next logits, and we loop
///      from 4 until a stop token, the token budget, or cancellation.
///
/// LEARNING NOTE - sharing the GPU with search
/// -------------------------------------------
/// MLX has a single GPU command stream, and this app also runs the embedder there for interactive
/// search. Every GPU touch in this class goes through `gate(_:)`, which routes to the embedder's
/// `OmniEngine` priority gate at "chat" priority: a user's search query (high priority) preempts
/// generation between tokens, while generation in turn outranks background indexing. Because we
/// acquire the gate per decode step, a search waits at most one token's worth of GPU time. Sampling
/// and detokenization happen on the CPU, outside the gate, so they never block search at all.
public final class ChatEngine: @unchecked Sendable {
    private let modelDir: URL
    private weak var gpuCoordinator: OmniEngine?

    private var config: ChatConfig?
    private var decoder: Qwen3ChatDecoder?
    private var tokenizer: (any Tokenizer)?

    private let stateLock = NSLock()
    private var loaded = false
    private var generating = false

    public init(modelDir: URL, gpuCoordinator: OmniEngine?) {
        self.modelDir = modelDir
        self.gpuCoordinator = gpuCoordinator
    }

    public var isLoaded: Bool { stateLock.withLock { loaded } }

    /// Route GPU work through the embedder's priority gate when present (chat priority), else run it
    /// directly (headless tools/tests with no embedder loaded).
    private func gate<T>(_ work: () -> T) -> T {
        if let c = gpuCoordinator { return c.runChatGPU(work) }
        return work()
    }

    // MARK: - Lifecycle

    /// Load weights + tokenizer. Idempotent. Cheap: weights are memory-mapped lazily, so the real
    /// ~1 GB read happens on the first prefill, not here.
    public func load() async throws {
        if isLoaded { return }
        let cfg = try ChatConfig(modelDir: modelDir)
        let weights = try ChatWeightStore(modelDir: modelDir, config: cfg)
        let dec = Qwen3ChatDecoder(weights: weights, config: cfg)
        let tok = try await AutoTokenizer.from(directory: modelDir)
        stateLock.withLock {
            self.config = cfg
            self.decoder = dec
            self.tokenizer = tok
            self.loaded = true
        }
    }

    /// Drop the model so MLX can reclaim its buffers. Safe to call any time; a later `load()` reloads.
    public func unload() {
        stateLock.withLock {
            decoder = nil
            tokenizer = nil
            config = nil
            loaded = false
        }
        MLX.GPU.clearCache()
    }

    // MARK: - Streaming generation

    /// Render -> tokenize -> prefill -> stream tokens. The returned stream yields UTF-8-safe text
    /// chunks and finishes on a stop token, `params.maxTokens`, or cancellation (terminate the
    /// stream or cancel the enclosing Task). `onStats` fires once at the end with timing.
    public func generate(messages: [ChatMessage],
                         params: ChatGenerationParams,
                         onStats: (@Sendable (ChatGenerationStats) -> Void)? = nil)
        -> AsyncThrowingStream<String, Error>
    {
        AsyncThrowingStream { continuation in
            // Enforce one generation at a time.
            let ok: Bool = self.stateLock.withLock {
                guard self.loaded else { return false }
                if self.generating { return false }
                self.generating = true
                return true
            }
            guard ok else {
                continuation.finish(throwing: OmniError.model(
                    self.isLoaded ? "chat generation already running" : "chat model not loaded"))
                return
            }

            let cancelled = CancelFlag()
            continuation.onTermination = { _ in cancelled.set() }

            let task = Task.detached(priority: .userInitiated) {
                defer { self.stateLock.withLock { self.generating = false } }
                do {
                    try self.runGeneration(messages: messages, params: params,
                                           cancelled: cancelled, continuation: continuation,
                                           onStats: onStats)
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in cancelled.set(); task.cancel() }
        }
    }

    private func runGeneration(messages: [ChatMessage],
                               params: ChatGenerationParams,
                               cancelled: CancelFlag,
                               continuation: AsyncThrowingStream<String, Error>.Continuation,
                               onStats: (@Sendable (ChatGenerationStats) -> Void)?) throws {
        guard let decoder = decoder, let tokenizer = tokenizer, let cfg = config else {
            throw OmniError.model("chat model not loaded")
        }

        let trimmed = trimToContext(messages, tokenizer: tokenizer, cfg: cfg, reserve: params.maxTokens)
        let prompt = ChatTemplate.render(trimmed, addGenerationPrompt: true)
        let promptIds = try tokenizer.encode(text: prompt, addSpecialTokens: false)

        var cache = decoder.makeCache()
        let detok = Detokenizer(tokenizer: tokenizer)
        var sampler = Sampler(temperature: params.temperature, topP: params.topP,
                              seed: params.seed ?? 0x9E37_79B9_7F4A_7C15)

        // Prefill: one gate acquisition for the whole prompt.
        let prefillStart = Date()
        var logits = gate { () -> [Float] in
            let l = decoder.prefill(promptIds, cache: cache)
            eval(l)
            return l.asArray(Float.self)
        }
        let prefillSeconds = -prefillStart.timeIntervalSinceNow

        // NaN insurance: a cold MLX device has very rarely produced a non-finite first forward
        // (observed on the embedder). Reload once and retry before giving up.
        if !logits.allSatisfy({ $0.isFinite }) {
            reloadWeights()
            cache = self.decoder?.makeCache() ?? cache
            logits = gate { () -> [Float] in
                let l = (self.decoder ?? decoder).prefill(promptIds, cache: cache)
                eval(l)
                return l.asArray(Float.self)
            }
        }

        let liveDecoder = self.decoder ?? decoder
        let decodeStart = Date()
        var generated = 0
        var token = sampler.sample(logits)
        while generated < params.maxTokens {
            if cancelled.isSet || Task.isCancelled { break }
            if cfg.eosTokenIds.contains(token) { break }
            if let chunk = try detok.consume(token) { continuation.yield(chunk) }
            generated += 1
            if generated >= params.maxTokens { break }

            // One decode step = one gate acquisition; search can preempt between tokens.
            let host = gate { () -> [Float] in
                let l = liveDecoder.step(token, cache: cache)
                eval(l)
                return l.asArray(Float.self)
            }
            token = sampler.sample(host)   // sample on the CPU, outside the gate
        }

        onStats?(ChatGenerationStats(promptTokens: promptIds.count, generatedTokens: generated,
                                     prefillSeconds: prefillSeconds,
                                     decodeSeconds: -decodeStart.timeIntervalSinceNow))
    }

    /// Drop oldest non-system messages until the rendered prompt fits in the context window with room
    /// for `reserve` output tokens. Last resort, this can drop the newest user message too; that is
    /// unreachable today because ChatContextBuilder's charBudget (12k chars, ~3-4k tokens) keeps a
    /// single RAG turn well under the post-reserve budget - keep those two bounds in step.
    private func trimToContext(_ messages: [ChatMessage], tokenizer: any Tokenizer,
                               cfg: ChatConfig, reserve: Int) -> [ChatMessage] {
        let budget = max(256, cfg.maxContextTokens - reserve)
        var msgs = messages
        func tokenCount(_ m: [ChatMessage]) -> Int {
            ((try? tokenizer.encode(text: ChatTemplate.render(m), addSpecialTokens: false)) ?? []).count
        }
        while tokenCount(msgs) > budget {
            // Find the first non-system message to drop.
            guard let idx = msgs.firstIndex(where: { $0.role != .system }) else { break }
            msgs.remove(at: idx)
            if msgs.allSatisfy({ $0.role == .system }) { break }
        }
        return msgs
    }

    private func reloadWeights() {
        guard let cfg = config,
              let weights = try? ChatWeightStore(modelDir: modelDir, config: cfg) else { return }
        let dec = Qwen3ChatDecoder(weights: weights, config: cfg)
        stateLock.withLock { self.decoder = dec }
    }

    // MARK: - Verification primitives (synchronous; used by omni-verify and tests)

    /// Token ids for the rendered prompt of `messages` (with the generation prompt appended).
    public func promptTokenIds(messages: [ChatMessage]) throws -> [Int] {
        guard let tokenizer = tokenizer else { throw OmniError.model("chat model not loaded") }
        let prompt = ChatTemplate.render(messages, addGenerationPrompt: true)
        return try tokenizer.encode(text: prompt, addSpecialTokens: false)
    }

    /// fp32 logits for the last position after prefilling `tokenIds`.
    public func prefillLogits(tokenIds: [Int]) throws -> [Float] {
        guard let decoder = decoder else { throw OmniError.model("chat model not loaded") }
        let cache = decoder.makeCache()
        return gate { () -> [Float] in
            let l = decoder.prefill(tokenIds, cache: cache)
            eval(l)
            return l.asArray(Float.self)
        }
    }

    /// Greedy (argmax) continuation of `tokenIds` for `steps` tokens, validating the KV-cache decode
    /// path against a reference. Returns the generated token ids in order.
    public func greedyContinuation(tokenIds: [Int], steps: Int) throws -> [Int] {
        guard let decoder = decoder else { throw OmniError.model("chat model not loaded") }
        let cache = decoder.makeCache()
        var out = [Int]()
        var logits = gate { () -> [Float] in
            let l = decoder.prefill(tokenIds, cache: cache)
            eval(l)
            return l.asArray(Float.self)
        }
        for _ in 0 ..< steps {
            let next = argmax(logits)
            out.append(next)
            logits = gate { () -> [Float] in
                let l = decoder.step(next, cache: cache)
                eval(l)
                return l.asArray(Float.self)
            }
        }
        return out
    }

    /// Last-position logits computed the INCREMENTAL way: prefill the first token, then decode the
    /// rest one at a time through the KV cache. Equals `prefillLogits` when the cache is correct, so
    /// the two together are a self-contained KV-cache check (no Python needed).
    public func incrementalPrefillLogits(tokenIds: [Int]) throws -> [Float] {
        guard let decoder = decoder, !tokenIds.isEmpty else { throw OmniError.model("chat model not loaded") }
        let cache = decoder.makeCache()
        var logits = gate { () -> [Float] in let l = decoder.prefill([tokenIds[0]], cache: cache); eval(l); return l.asArray(Float.self) }
        for i in 1 ..< tokenIds.count {
            logits = gate { () -> [Float] in let l = decoder.step(tokenIds[i], cache: cache); eval(l); return l.asArray(Float.self) }
        }
        return logits
    }

    /// Tokenize text the way generation does (the model's added-vocab special tokens are recognized).
    public func encodeText(_ text: String) throws -> [Int] {
        guard let tokenizer = tokenizer else { throw OmniError.model("chat model not loaded") }
        return try tokenizer.encode(text: text, addSpecialTokens: false)
    }

    /// One-shot decode of token ids to text.
    public func decodeTokens(_ ids: [Int]) throws -> String {
        guard let tokenizer = tokenizer else { throw OmniError.model("chat model not loaded") }
        return try tokenizer.decode(tokenIds: ids, skipSpecialTokens: true)
    }

    /// Streaming decode of token ids (feeds them through the incremental ``Detokenizer``).
    public func streamDecodeTokens(_ ids: [Int]) throws -> String {
        guard let tokenizer = tokenizer else { throw OmniError.model("chat model not loaded") }
        let detok = Detokenizer(tokenizer: tokenizer)
        var out = ""
        for id in ids { if let chunk = try detok.consume(id) { out += chunk } }
        return out
    }

    private func argmax(_ logits: [Float]) -> Int {
        var best = 0, bestV = logits[0]
        for i in 1 ..< logits.count where logits[i] > bestV { bestV = logits[i]; best = i }
        return best
    }
}

/// Thread-safe one-shot cancellation flag shared between the stream's onTermination and the worker.
final class CancelFlag: @unchecked Sendable {
    private let lock = NSLock()
    private var flag = false
    var isSet: Bool { lock.withLock { flag } }
    func set() { lock.withLock { flag = true } }
}
