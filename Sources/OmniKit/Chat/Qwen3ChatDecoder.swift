import Foundation
import MLX
import MLXFast
import MLXNN

/// The Qwen3-1.7B causal language model, implemented by hand on MLX primitives.
///
/// This is a deliberate twin of `Qwen3Backbone` (the embedder's transformer). They share the SAME
/// architecture - that is the whole reason this model was chosen - but they are kept as separate
/// files on purpose: the embedder is numerically frozen against a Python reference, while this
/// decoder adds the three things generation needs and embedding does not:
///   1. quantized (4-bit) weights instead of fp32/bf16 (``QuantizedTensor``),
///   2. a per-layer KV cache so decoding is one forward pass per token (``KVCache``),
///   3. a tied output head that turns the final hidden state into a logit per vocabulary token.
///
/// LEARNING NOTE - one transformer block, end to end
/// -------------------------------------------------
/// Each of the 28 blocks does, in order:
///   h -> RMSNorm -> attention -> add back to h     (the "attention residual")
///   h -> RMSNorm -> SwiGLU MLP -> add back to h     (the "MLP residual")
/// Attention: project the normalized h into queries/keys/values, give each head its own RMSNorm
/// (a Qwen3 detail, `use_qk_norm`), rotate q and k by their position (RoPE), append k/v to the cache,
/// then `softmax(q . k^T / sqrt(headDim)) . v`. The MLP is "SwiGLU": `down( silu(gate(h)) * up(h) )`.
/// After all blocks, a final RMSNorm, then the tied embedding matrix maps the last hidden vector to
/// `[vocab]` logits.
///
/// LEARNING NOTE - precision
/// -------------------------
/// Matmuls and attention run in bf16 (the weights are 4-bit, so an fp32 compute path would be
/// pointless and just slower). RMSNorm computes its variance in fp32 and casts back - the same idiom
/// the embedder uses - because the mean-of-squares is the one place bf16's small mantissa actually
/// hurts. Logits are returned as fp32 so the CPU-side sampler sees clean numbers.
final class Qwen3ChatDecoder {
    let cfg: ChatConfig
    private let w: ChatWeightStore
    private let rope: RoPE
    private let scale: Float
    private let embedTokens: QuantizedTensor

    init(weights: ChatWeightStore, config: ChatConfig) {
        self.cfg = config
        self.w = weights
        self.rope = RoPE(dimensions: config.headDim, traditional: false, base: config.ropeTheta)
        self.scale = Float(pow(Double(config.headDim), -0.5))
        self.embedTokens = weights.quantized("model.embed_tokens")
    }

    /// One empty KV cache per layer.
    func makeCache() -> [KVCache] {
        (0 ..< cfg.numLayers).map { _ in KVCache(kvHeads: cfg.numKVHeads, headDim: cfg.headDim) }
    }

    /// Prefill: run the whole prompt through the model in one causal pass, filling `cache`, and
    /// return the UNEVALUATED logits `[1, vocab]` (fp32) for the last prompt position - the
    /// distribution to sample the first generated token from. The caller drives `eval`.
    func prefill(_ tokenIds: [Int], cache: [KVCache]) -> MLXArray {
        forward(tokenIds, cache: cache, causal: true)
    }

    /// Decode one step: run a single token at RoPE offset `cache[0].offset`, append its k/v, and
    /// return the UNEVALUATED logits `[1, vocab]` (fp32). No mask is needed (the one query may attend
    /// to every cached position; there are no future positions).
    func step(_ tokenId: Int, cache: [KVCache]) -> MLXArray {
        forward([tokenId], cache: cache, causal: false)
    }

    // MARK: - Shared forward pass

    private func forward(_ tokenIds: [Int], cache: [KVCache], causal: Bool) -> MLXArray {
        let offset = cache[0].offset
        let ids = MLXArray(tokenIds.map { Int32($0) })
        var h = embedTokens.gatherRows(ids).reshaped([1, tokenIds.count, cfg.hiddenSize])

        let maskMode: MLXFast.ScaledDotProductAttentionMaskMode = causal ? .causal : .none
        for i in 0 ..< cfg.numLayers {
            let p = "model.layers.\(i)."
            h = h + attention(rmsNorm(h, p + "input_layernorm.weight"), p, cache: cache[i],
                              offset: offset, mask: maskMode)
            h = h + mlp(rmsNorm(h, p + "post_attention_layernorm.weight"), p)
        }
        h = rmsNorm(h, "model.norm.weight")

        // Only the last position feeds the output head, so slice BEFORE the big [vocab] matmul.
        let last = h[0, tokenIds.count - 1].reshaped([1, cfg.hiddenSize])   // [1, hidden]
        return embedTokens.matmul(last).asType(.float32)                    // [1, vocab]
    }

    private func attention(_ x: MLXArray, _ p: String, cache: KVCache, offset: Int,
                           mask: MLXFast.ScaledDotProductAttentionMaskMode) -> MLXArray {
        let L = x.dim(1)
        var q = w.quantized(p + "self_attn.q_proj").matmul(x)
            .reshaped([1, L, cfg.numHeads, cfg.headDim]).transposed(0, 2, 1, 3)
        var k = w.quantized(p + "self_attn.k_proj").matmul(x)
            .reshaped([1, L, cfg.numKVHeads, cfg.headDim]).transposed(0, 2, 1, 3)
        let v = w.quantized(p + "self_attn.v_proj").matmul(x)
            .reshaped([1, L, cfg.numKVHeads, cfg.headDim]).transposed(0, 2, 1, 3)

        if cfg.useQKNorm {
            q = headNorm(q, p + "self_attn.q_norm.weight")
            k = headNorm(k, p + "self_attn.k_norm.weight")
        }
        q = rope(q, offset: offset)
        k = rope(k, offset: offset)

        // Append this step's k/v and attend over the full valid prefix. GQA (8 kv heads vs 16 q
        // heads) is handled inside scaledDotProductAttention by broadcasting kv heads across queries.
        let (ck, cv) = cache.update(keys: k, values: v)
        var o = MLXFast.scaledDotProductAttention(queries: q, keys: ck, values: cv, scale: scale, mask: mask)
        o = o.transposed(0, 2, 1, 3).reshaped([1, L, cfg.numHeads * cfg.headDim])
        return w.quantized(p + "self_attn.o_proj").matmul(o)
    }

    private func mlp(_ x: MLXArray, _ p: String) -> MLXArray {
        let gate = MLXNN.silu(w.quantized(p + "mlp.gate_proj").matmul(x))
        let up = w.quantized(p + "mlp.up_proj").matmul(x)
        return w.quantized(p + "mlp.down_proj").matmul(gate * up)
    }

    /// RMSNorm over the last (feature) axis, using MLX's fused kernel. This is the exact same op the
    /// reference (mlx_lm's nn.RMSNorm == mx.fast.rms_norm) uses, including its internal fp32
    /// accumulation, so the Swift logits match the Python reference to within bf16 rounding.
    private func rmsNorm(_ x: MLXArray, _ key: String) -> MLXArray {
        MLXFast.rmsNorm(x, weight: w.plain(key), eps: cfg.rmsNormEps)
    }

    /// Per-head RMSNorm (same fused kernel; normalizes over head_dim, the last axis).
    private func headNorm(_ x: MLXArray, _ key: String) -> MLXArray {
        MLXFast.rmsNorm(x, weight: w.plain(key), eps: cfg.rmsNormEps)
    }
}
