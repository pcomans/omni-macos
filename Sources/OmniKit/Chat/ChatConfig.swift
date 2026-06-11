import Foundation

/// Static description of the chat language model (Qwen3-1.7B), parsed from its config.json.
///
/// LEARNING NOTE - what a causal-LM config is
/// ------------------------------------------
/// A decoder-only language model is a stack of identical transformer blocks. To build it you only
/// need a handful of numbers: how wide each token's vector is (`hiddenSize`), how many blocks are
/// stacked (`numLayers`), and the shape of the attention and MLP inside each block. Everything else
/// in this file is one of those numbers, read straight from the model's config.json so the Swift
/// code and the published weights can never disagree.
///
/// LEARNING NOTE - grouped-query attention (GQA)
/// ---------------------------------------------
/// Plain multi-head attention gives every query head its own key/value head. That is a lot of
/// key/value data to compute and (crucially for generation) to keep around in the KV cache. GQA
/// keeps the full number of *query* heads (`numHeads`, 16 here) but uses far fewer *key/value*
/// heads (`numKVHeads`, 8 here); each KV head is shared by `numHeads / numKVHeads` query heads.
/// This halves the KV cache and the k/v projections at almost no quality cost, which is exactly the
/// trade you want for on-device decoding. `headDim` (128) is the width of a single head; note that
/// 16 * 128 = 2048 = `hiddenSize`, while k and v project to 8 * 128 = 1024.
///
/// LEARNING NOTE - tied embeddings
/// -------------------------------
/// `tieWordEmbeddings` means the model reuses the token-embedding matrix as the output ("unembed")
/// head: the same `[vocab, hidden]` matrix that turns a token id into a vector also turns the final
/// hidden vector back into a score per vocabulary token. So there is no separate `lm_head` weight in
/// the checkpoint (we verified: the safetensors has no `lm_head.*` key). `Qwen3ChatDecoder` applies
/// the embedding matrix a second time, transposed, to produce logits.
public struct ChatConfig: Sendable {
    public var hiddenSize = 2048
    public var numLayers = 28
    public var intermediateSize = 6144
    public var numHeads = 16
    public var numKVHeads = 8
    public var headDim = 128
    public var rmsNormEps: Float = 1e-6
    public var vocabSize = 151936
    public var ropeTheta: Float = 1_000_000
    public var tieWordEmbeddings = true
    public var useQKNorm = true

    /// Group size and bit width of the 4-bit affine quantization (see ``QuantizedTensor``).
    public var quantGroupSize = 128
    public var quantBits = 4

    /// Stop tokens: `<|im_end|>` ends an assistant turn; `<|endoftext|>` is a safety stop.
    public var eosTokenIds: [Int] = [151645, 151643]

    /// App-imposed context cap (NOT the model's 65536 limit). Bounds the KV cache memory and the
    /// prompt length we will accept. The KV cache grows on demand in slabs, so this is just a ceiling,
    /// not a preallocation; ~8192 tokens lets a dozen retrieved chunks fit alongside the answer.
    public var maxContextTokens = 8192

    public init() {}

    /// Parse from the model directory's config.json. Missing keys keep the defaults above, which are
    /// the verified Qwen3-1.7B values, so a slightly different conversion still loads sensibly.
    public init(modelDir: URL) throws {
        var cfg = ChatConfig()
        let url = modelDir.appendingPathComponent("config.json")
        guard let data = try? Data(contentsOf: url),
              let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else {
            throw OmniError.model("chat config.json not found or unreadable at \(url.path)")
        }
        cfg.hiddenSize = root["hidden_size"] as? Int ?? cfg.hiddenSize
        cfg.numLayers = root["num_hidden_layers"] as? Int ?? cfg.numLayers
        cfg.intermediateSize = root["intermediate_size"] as? Int ?? cfg.intermediateSize
        cfg.numHeads = root["num_attention_heads"] as? Int ?? cfg.numHeads
        cfg.numKVHeads = root["num_key_value_heads"] as? Int ?? cfg.numKVHeads
        cfg.headDim = root["head_dim"] as? Int ?? cfg.headDim
        cfg.vocabSize = root["vocab_size"] as? Int ?? cfg.vocabSize
        if let eps = root["rms_norm_eps"] as? Double { cfg.rmsNormEps = Float(eps) }
        if let theta = root["rope_theta"] as? Double { cfg.ropeTheta = Float(theta) }
        cfg.tieWordEmbeddings = root["tie_word_embeddings"] as? Bool ?? cfg.tieWordEmbeddings
        cfg.useQKNorm = root["use_qk_norm"] as? Bool ?? cfg.useQKNorm
        if let q = root["quantization"] as? [String: Any] {
            cfg.quantGroupSize = q["group_size"] as? Int ?? cfg.quantGroupSize
            cfg.quantBits = q["bits"] as? Int ?? cfg.quantBits
        }
        if let eos = root["eos_token_id"] as? Int, !cfg.eosTokenIds.contains(eos) {
            cfg.eosTokenIds.insert(eos, at: 0)
        }
        self = cfg
    }
}
