# How the local chat works: LLM inference from scratch

This document walks the full path a chat answer travels, from a question you type to words streaming
back, and points at the Swift file that implements each step. The chat model is **Qwen3-1.7B** in
4-bit, run entirely on-device through MLX. It was chosen because its architecture is identical to the
Qwen3 transformer the search embedder already implements by hand (`Qwen3Backbone.swift`), so the
generation path is a focused addition rather than a second model framework.

Everything here is in `Sources/OmniKit/Chat/`. The files are intentionally small and single-purpose so
each idea can be read on its own.

## 1. The shape of the problem

A chat model is a function from "the conversation so far" to "a probability distribution over the next
token." You sample one token from that distribution, append it, and ask again. Repeat until the model
emits a stop token. So generating text is a loop, and almost all the engineering is about making each
turn of that loop fast and the text coming out of it clean.

The pipeline, end to end (`ChatEngine.swift` orchestrates it):

1. **Render** the messages into the exact string format the model was trained on - `ChatTemplate.swift`
2. **Tokenize** that string into integer ids - the tokenizer (swift-tokenizers)
3. **Prefill**: run the whole prompt through the model once, building the KV cache and producing the
   logits for the first new token - `Qwen3ChatDecoder.swift` + `KVCache.swift`
4. **Sample** a token from those logits - `Sampler.swift`
5. **Detokenize** that token id into displayable text - `Detokenizer.swift`
6. **Decode step**: run that one token to get the next logits, and loop to 4 - `Qwen3ChatDecoder.swift`

## 2. Tokens and the chat template

Models do not see text; they see integers. A tokenizer maps text to a sequence of ids drawn from a
fixed vocabulary (~152k entries for Qwen3). Qwen3 uses byte-level BPE, which matters later for
streaming (section 7).

A *base* model only continues text. To make it behave like an assistant, training wraps each turn in
marker tokens, and inference has to reproduce that wrapping exactly. Qwen3's format is "ChatML":

```
<|im_start|>system
{system instruction}<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
```

That trailing open `assistant` turn is the cue for the model to generate. One Qwen3 detail: it is a
"hybrid thinking" model that by default emits a `<think>...</think>` reasoning block first. For grounded
folder Q&A we want the answer directly, so we prefill an already-closed, empty think block
(`<think>\n\n</think>\n\n`) into the assistant turn - the documented non-thinking convention. Because
that block is just input tokens, it works no matter which template the checkpoint shipped with. See
`ChatTemplate.swift`; the fixture generator (`Tools/gen_chat_fixtures.py`) proves our hardcoded turn
formatting matches the reference template byte-for-byte.

## 3. Four-bit weights

Qwen3-1.7B has ~1.7 billion parameters. In bf16 that is ~3.4 GB, and generating each token reads every
weight once, so speed is bound by memory bandwidth. 4-bit quantization stores each weight in half a
byte, cutting footprint and bandwidth ~4x.

MLX uses **affine group quantization**: each stored 4-bit integer `q` maps back to a real value by
`w = scale * q + bias`, and a group of 128 consecutive weights shares one scale/bias pair. So a weight
matrix of logical shape `[out, in]` is stored as three tensors:

- `weight`: `[out, in/8]` UInt32, packing 8 four-bit values per 32-bit word
- `scales`: `[out, in/128]`
- `biases`: `[out, in/128]`

The important trick (`QuantizedTensor.swift`): we never fully unpack the matrix to multiply it.
`MLX.quantizedMM` is a fused kernel that reads the packed 4-bit data and dequantizes each group on the
fly inside the matmul, so the only large thing crossing the memory bus stays 4-bit. We fully dequantize
only for the embedding lookup, where we touch a few rows, not the whole table. `ChatWeightStore.swift`
loads the `.weight`/`.scales`/`.biases` triplets (memory-mapped, so the ~1 GB read is deferred to the
first forward pass).

## 4. One transformer step

`Qwen3ChatDecoder.swift` is 28 stacked identical blocks. Each block does, with residual adds:

```
h = h + attention(rmsnorm(h))
h = h + swiglu_mlp(rmsnorm(h))
```

- **RMSNorm** rescales a vector to unit root-mean-square and multiplies by a learned gain. We compute
  the mean-of-squares in fp32 (the one place bf16's short mantissa hurts) and cast back.
- **Attention** projects the normalized hidden state into queries, keys, and values; gives each head its
  own RMSNorm (a Qwen3 feature, `use_qk_norm`); rotates q and k by their position with RoPE; then
  computes `softmax(q . k^T / sqrt(headDim)) . v`. This model uses **grouped-query attention**: 16 query
  heads but only 8 key/value heads (each KV head serves two query heads), which halves the KV cache.
- **SwiGLU MLP** is `down( silu(gate(h)) * up(h) )`.

After the blocks, a final RMSNorm, then the **tied output head**: the same embedding matrix that turned
token ids into vectors is applied transposed to turn the final hidden vector into a logit per
vocabulary token. (Qwen3-1.7B ties these weights, so there is no separate `lm_head` in the checkpoint.)

## 5. The KV cache

This is the single most important idea for fast generation. Attention lets each token attend to every
earlier token. Naively, generating token `t` reruns the model over all `t` positions, so producing `N`
tokens is `O(N^2)` work. But when you append a token, the keys and values of all *earlier* tokens do
not change - only the new token adds a K/V row and issues a query.

So we remember each layer's K and V (`KVCache.swift`). Then:

- **Prefill** runs the whole prompt once, fills the cache for every prompt position, and returns the
  last position's logits. It needs a **causal mask** so a prompt token cannot peek at later ones.
- **Decode** runs one token per step, appends its single K/V row, and attends over the whole cache. It
  needs **no mask** - the lone new query may see all cached positions and there are no future ones.

That turns each step into one forward pass over a single token: `O(t)` math, but crucially one pass
instead of `t`. The cache costs `2 * numLayers * numKVHeads * headDim * bytes` per token; for this model
in bf16 that is ~112 KiB/token, so the 4096-token context cap is ~448 MiB. The buffer grows in 256-token
slabs to avoid reallocating every step, and `offset` (the number of valid positions) doubles as the
RoPE position for the next token.

## 6. Choosing the next token

The model gives a logit (unnormalized score) per vocabulary token. `Sampler.swift` turns that into a
choice:

- **Temperature** divides the logits before softmax. `T < 1` sharpens (more confident, more
  repetitive); `T > 1` flattens (more random); `T = 0` means "take the single highest logit" (greedy),
  which is deterministic and what the parity fixtures use.
- **Softmax** turns logits into probabilities: `p_i = exp(logit_i) / sum_j exp(logit_j)`.
- **Top-p (nucleus)** keeps the smallest set of most-likely tokens whose probabilities sum to at least
  `p`, zeros the rest, renormalizes, and samples from that. This keeps the model creative among
  plausible continuations but almost never lets it derail into nonsense.

Two practical choices: we only softmax the top ~512 logits (the rest hold negligible mass, and a full
152k-way softmax per token on the CPU is wasteful), and we sample on the CPU with a seedable generator
(`SplitMix64`) so runs are reproducible and the GPU stream stays free for search.

## 7. Streaming text out

You cannot just decode each token id to text and concatenate. Byte-level BPE tokens are chunks of
UTF-8 *bytes*, and a single character (an emoji, a CJK glyph) can span several tokens. Decoding a token
that ends mid-character yields the replacement character "?". The fix is to decode the running list and
emit only the new, complete suffix, holding back any trailing incomplete scalar until the next token
finishes it. swift-tokenizers ships this as `StreamingDetokenizer`; `Detokenizer.swift` wraps it.

## 8. Sharing one GPU

MLX has a single GPU command stream, and this app also runs the search embedder on it. If generation
hogged the GPU, search would stall mid-answer. `OmniEngine.swift` has a three-tier priority gate:

- **high** - interactive search queries
- **chat** - LLM generation
- **low** - background indexing and the folder-map projection

A high-priority op preempts both others; chat preempts indexing. Each tier waits at most one in-flight
op. `ChatEngine` acquires the gate once per decode step (via `runChatGPU`), so a search waits at most one
token's worth of GPU time before it runs. Sampling and detokenization happen on the CPU, outside the
gate, so they never block search at all.

## 9. Grounding answers in your files (RAG)

The chat does not answer from the model's memory; it answers from your indexed files. For each question
(`ChatContextBuilder.swift`):

1. Embed the question with the existing `OmniTextEncoder` (the same model that powers search), at the
   high-priority tier.
2. Search the vector store, restricted to the selected folder, for the most similar chunks.
3. Re-derive each chunk's full text from the original file. The index stores only short snippets, so we
   re-extract and re-chunk the file with the shared `TextChunker.swift` (the exact logic the indexer
   uses). A snippet-equality check guards against the chunk setting having changed since indexing; on
   any mismatch we fall back to the stored snippet, so a citation can never point at the wrong text.
4. Assemble a prompt with numbered sources and an instruction to answer only from them and cite inline,
   then generate.

The cited file names become clickable chips in the UI, so an answer links straight back to the source.

## 10. Trust, but verify

None of this is assumed correct - it is measured against the Python reference, the same discipline the
embedder uses. `Tools/gen_chat_fixtures.py` runs the official model through `mlx_lm` and records, for a
fixed prompt: the rendered string, the exact token ids, the full fp32 logits after prefill, and 32
greedy continuation tokens (plus the top1-top2 logit gap at each step). `omni-verify chatverify` then
reproduces all of it from the Swift decoder and checks:

1. the template renders the identical string,
2. the prompt tokenizes to the identical ids,
3. the prefill logits match to cosine >= 0.999,
4. the greedy continuation matches token-for-token, allowing a divergence only where the reference logit
   gap was a true sub-`1e-3` bf16 tie.

Cosine on the logits proves the forward pass is faithful; the greedy match proves the KV-cache decode
path is faithful too. Run it with `./Scripts/run-verify.sh chatverify <chatModelDir> Fixtures/chat_fixtures.json`.
(`run-verify.sh` builds through xcodebuild because MLX's Metal kernels are produced by an Xcode build
step that the plain `swift run` CLI does not invoke.)
```
