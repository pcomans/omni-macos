import Foundation
import MLX

/// Loads the Qwen3-1.7B 4-bit chat weights from `model.safetensors` and hands them out as either
/// quantized matrices (``QuantizedTensor``) or plain norm vectors, keyed by their stored name.
///
/// LEARNING NOTE - why this is separate from `WeightStore`
/// ------------------------------------------------------
/// The embedder's `WeightStore` does three things this model must NOT do: it merges a retrieval
/// LoRA, it upcasts the backbone to fp32, and it keys everything under `language_model.*`. None of
/// that applies to a stock causal LM. Keeping a separate, dead-simple loader here means the embedder
/// path (which is numerically verified against a Python reference) is never touched by chat work.
///
/// LEARNING NOTE - safetensors and lazy memory mapping
/// ---------------------------------------------------
/// `model.safetensors` is just a header (a JSON map of name -> dtype/shape/byte-range) followed by
/// the raw tensor bytes. `MLX.loadArrays` memory-maps the file: the `MLXArray`s it returns are lazy
/// handles, and the bytes are not actually paged in from disk until the first computation that needs
/// them is `eval`-ed. That is why constructing this store is cheap (~milliseconds) and the real
/// ~1 GB read happens on the first `prefill`. The quantized weights stay packed (UInt32); the norm
/// weights are small bf16 vectors stored in full precision because a per-channel norm has no groups
/// to quantize and the savings would be negligible.
public struct ChatWeightStore {
    let raw: [String: MLXArray]
    let groupSize: Int
    let bits: Int

    public init(modelDir: URL, config: ChatConfig) throws {
        let url = modelDir.appendingPathComponent("model.safetensors")
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw OmniError.model("chat model.safetensors not found at \(url.path)")
        }
        self.raw = try loadArrays(url: url)
        self.groupSize = config.quantGroupSize
        self.bits = config.quantBits
        // Sanity check one expected key so a wrong/corrupt checkpoint fails loudly at load, not deep
        // inside the first forward pass.
        guard raw["model.embed_tokens.weight"] != nil, raw["model.norm.weight"] != nil else {
            throw OmniError.model("chat weights missing expected keys (model.embed_tokens / model.norm)")
        }
    }

    func has(_ key: String) -> Bool { raw[key] != nil }

    /// The quantized triplet (`.weight`/`.scales`/`.biases`) at `base`, e.g.
    /// "model.layers.0.self_attn.q_proj". Throws via fatalError on a missing key because every key
    /// is known statically from the architecture; a miss means a wrong checkpoint, not a runtime case.
    func quantized(_ base: String) -> QuantizedTensor {
        guard let w = raw["\(base).weight"], let s = raw["\(base).scales"], let b = raw["\(base).biases"] else {
            fatalError("chat weights: missing quantized triplet for \(base)")
        }
        return QuantizedTensor(weight: w, scales: s, biases: b, groupSize: groupSize, bits: bits)
    }

    /// A plain (non-quantized) weight such as an RMSNorm scale vector.
    func plain(_ key: String) -> MLXArray {
        guard let v = raw[key] else { fatalError("chat weights: missing plain weight \(key)") }
        return v
    }
}
