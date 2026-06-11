import Foundation
import MLX

/// A single 4-bit quantized weight matrix: packed integer data plus the per-group scales and biases
/// needed to turn it back into real numbers.
///
/// LEARNING NOTE - why quantize at all
/// -----------------------------------
/// Qwen3-1.7B has ~1.7 billion parameters. In bf16 (2 bytes each) that is ~3.4 GB; in fp32, ~6.8 GB.
/// On-device that is a lot of memory and, more importantly, a lot of memory *bandwidth*: generating
/// one token reads every weight once, so token speed is bound by how fast you can stream the weights
/// from RAM. 4-bit quantization stores each weight in half a byte, cutting both the footprint and the
/// bandwidth by ~4x versus bf16. The published Qwen3-1.7B-MLX-4bit checkpoint is already in this form.
///
/// LEARNING NOTE - affine group quantization (the exact scheme MLX uses)
/// --------------------------------------------------------------------
/// "Affine" means each stored 4-bit integer `q` (0...15) maps back to a real value by an affine
/// formula `w = scale * q + bias`. A single scale/bias pair for a whole matrix would be far too
/// coarse, so MLX uses *group* quantization: consecutive runs of `groupSize` (128 here) weights
/// along the input dimension share one scale and one bias. So for a weight matrix of logical shape
/// `[out, in]`:
///   - `weight`  is `[out, in / 8]` of UInt32. Each UInt32 packs 8 four-bit values (8 * 4 = 32 bits).
///   - `scales`  is `[out, in / 128]`, one per group.
///   - `biases`  is `[out, in / 128]`, one per group.
/// To reconstruct `w[o, i]`: unpack the nibble at column `i`, then apply the scale/bias of its group
/// `i / 128`.
///
/// LEARNING NOTE - why we never fully unpack for a matmul
/// -----------------------------------------------------
/// The naive approach would dequantize the whole `[out, in]` matrix to bf16 and then matmul. That
/// throws away the entire bandwidth win - you would touch the full-size matrix anyway. Instead MLX
/// provides `quantizedMM`, a fused kernel that reads the *packed* weights and dequantizes each group
/// on the fly inside the matmul, so the only large thing crossing the memory bus is the 4-bit data.
/// That fused path is what makes 4-bit actually faster, not just smaller. We only ever fully
/// dequantize for the embedding lookup, where we touch a handful of rows, not the whole matrix.
struct QuantizedTensor {
    let weight: MLXArray   // UInt32 packed, [out, in * bits / 32]
    let scales: MLXArray   // [out, in / groupSize]
    let biases: MLXArray   // [out, in / groupSize]
    let groupSize: Int
    let bits: Int

    /// `x @ W^T` with in-kernel dequantization. `transpose: true` matches how Linear layers store
    /// their weight as `[out, in]`, so the kernel computes `x[..., in] @ W[out, in]^T -> [..., out]`.
    func matmul(_ x: MLXArray) -> MLXArray {
        quantizedMM(x, weight, scales: scales, biases: biases,
                    transpose: true, groupSize: groupSize, bits: bits)
    }

    /// Embedding-style row gather: dequantize ONLY the rows named by `ids` (one row per token), not
    /// the whole `[vocab, hidden]` table. `ids` is int32 of shape `[L]`; returns `[L, hidden]` bf16.
    /// This mirrors MLXNN.QuantizedEmbedding: index the packed rows, then dequantize that small slice.
    func gatherRows(_ ids: MLXArray) -> MLXArray {
        dequantized(weight[ids], scales: scales[ids], biases: biases[ids],
                    groupSize: groupSize, bits: bits, dtype: .bfloat16)
    }
}
