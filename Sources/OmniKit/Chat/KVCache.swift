import Foundation
import MLX

/// Per-layer key/value cache that makes autoregressive generation fast.
///
/// LEARNING NOTE - the problem the cache solves
/// --------------------------------------------
/// Attention lets each token look at every earlier token. If you generated text by re-running the
/// whole transformer over the entire sequence-so-far for every new token, step `t` would attend over
/// `t` positions, and producing `N` tokens would cost on the order of `1 + 2 + ... + N = O(N^2)`
/// full forward passes. That is hopelessly slow.
///
/// The key observation: when you append one token, the keys (K) and values (V) computed for all the
/// *earlier* tokens do not change. Only the new token contributes a new K and V row, and only the new
/// token issues a query. So if we remember every layer's K and V from the past, each generation step
/// is just: compute K/V/Q for the single new token, append its K/V to the remembered tensors, and run
/// attention of that one query against all cached keys. That is `O(t)` work for step `t` and, vitally,
/// exactly ONE forward pass per token instead of `t`. The remembered K/V per layer is this cache.
///
/// LEARNING NOTE - prefill vs decode
/// ---------------------------------
/// Generation has two phases. "Prefill" runs the whole prompt through the model in a single batched
/// pass: it fills the cache for all prompt positions at once and produces the logits for the last
/// prompt token (the first thing to sample). "Decode" then runs one token at a time, each step
/// appending a single K/V row. Prefill needs a causal mask (a prompt token must not peek at later
/// prompt tokens); decode needs no mask at all, because the single new query is allowed to see every
/// cached position and there are no future positions to hide.
///
/// LEARNING NOTE - memory
/// ----------------------
/// Each cached token costs, across all layers:
///     2 (K and V) * numLayers * numKVHeads * headDim * bytesPerElement
/// For Qwen3-1.7B in bf16 that is 2 * 28 * 8 * 128 * 2 = 114,688 bytes ~= 112 KiB per token, so a
/// full 4096-token context is ~448 MiB. We grow the backing buffer in fixed-size slabs (rather than
/// reallocating on every token) to keep allocation churn down; `offset` tracks how many positions are
/// actually valid, and doubles as the RoPE position for the next token.
final class KVCache {
    let kvHeads: Int
    let headDim: Int

    /// Number of valid cached positions. Also the RoPE offset to use for the next appended token.
    private(set) var offset = 0

    /// Grow capacity in slabs of this many tokens, so we reallocate ~once per 256 tokens, not per token.
    static let growthStep = 256

    private var keys: MLXArray?      // [1, kvHeads, capacity, headDim], bf16
    private var values: MLXArray?

    init(kvHeads: Int, headDim: Int) {
        self.kvHeads = kvHeads
        self.headDim = headDim
    }

    /// Append `newKeys`/`newValues` (shape `[1, kvHeads, L, headDim]`) and return views covering all
    /// valid positions so far (`[1, kvHeads, offset + L, headDim]`) for attention to read.
    func update(keys newKeys: MLXArray, values newValues: MLXArray) -> (keys: MLXArray, values: MLXArray) {
        let L = newKeys.dim(2)
        let needed = offset + L
        ensureCapacity(needed)
        // Write the new rows into the preallocated slab at [offset ..< offset+L] along the time axis.
        keys![0..., 0..., offset ..< needed, 0...] = newKeys
        values![0..., 0..., offset ..< needed, 0...] = newValues
        offset = needed
        // Return only the valid prefix; the slab may be larger.
        return (keys![0..., 0..., 0 ..< offset, 0...],
                values![0..., 0..., 0 ..< offset, 0...])
    }

    private func ensureCapacity(_ needed: Int) {
        let current = keys?.dim(2) ?? 0
        if needed <= current { return }
        let newCap = ((needed + Self.growthStep - 1) / Self.growthStep) * Self.growthStep
        let shape = [1, kvHeads, newCap, headDim]
        let grownK = MLXArray.zeros(shape, dtype: .bfloat16)
        let grownV = MLXArray.zeros(shape, dtype: .bfloat16)
        if let k = keys, let v = values, offset > 0 {
            grownK[0..., 0..., 0 ..< offset, 0...] = k[0..., 0..., 0 ..< offset, 0...]
            grownV[0..., 0..., 0 ..< offset, 0...] = v[0..., 0..., 0 ..< offset, 0...]
        }
        keys = grownK
        values = grownV
    }
}
