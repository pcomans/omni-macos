#!/usr/bin/env python3
"""Generate chat-decode parity fixtures for the hand-rolled Qwen3-1.7B Swift decoder.

Run (no venv needed; uv pulls mlx-lm into an ephemeral environment):

    uv run --with mlx-lm python Tools/gen_chat_fixtures.py

The model is taken from QWEN3_CHAT_MODEL_DIR if set, else downloaded from the Hub.

What it writes to Fixtures/chat_fixtures.json:
  - messages       : the fixture conversation (role/content), the exact thing Swift renders
  - prompt         : the rendered ChatML string (HF template, verified == our hardcoded renderer)
  - token_ids      : exact ids of `prompt`
  - prefill_logits : fp32 logits for the LAST prompt position (the first-token distribution)
  - greedy_tokens  : 32 greedy (argmax) continuation tokens
  - greedy_top2_gap: per-step (top1 - top2) logit gap, so the Swift gate can allow bf16 argmax ties

The Swift side (omni-verify chatverify) must reproduce all of these from the SAME 4-bit checkpoint.
"""

import json
import os
import sys

import mlx.core as mx
import numpy as np
from mlx_lm import load

# The fixture conversation. Content is arbitrary; it only has to be reproduced byte-for-byte by the
# Swift ChatTemplate. It mimics the RAG shape (a system instruction + a sources block + a question).
SYSTEM = (
    "You are a helpful assistant that answers questions about the user's local files.\n"
    "Answer using ONLY the numbered sources below. Cite sources inline with bracketed "
    "numbers like [1] or [2][3] after the statements they support. If the sources do not "
    "contain the answer, say you could not find it in the indexed files. Be concise."
)
USER = (
    "Sources:\n\n"
    "[1] notes.md (Line 1)\n"
    "The project deadline is March 14 and the total budget is 5000 dollars.\n\n"
    "Question: When is the project deadline?"
)
MESSAGES = [
    {"role": "system", "content": SYSTEM},
    {"role": "user", "content": USER},
]

EMPTY_THINK = "<think>\n\n</think>\n\n"


def hardcoded_render(messages, add_generation_prompt=True):
    """Must match ChatTemplate.render in Swift exactly."""
    out = ""
    for m in messages:
        out += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
    if add_generation_prompt:
        out += f"<|im_start|>assistant\n{EMPTY_THINK}"
    return out


def main():
    path = os.environ.get("QWEN3_CHAT_MODEL_DIR") or "Qwen/Qwen3-1.7B-MLX-4bit"
    print(f"loading {path}", file=sys.stderr)
    model, tokenizer = load(path)

    # 1. Validate the TURN FORMATTING against the reference template (the part that must match what
    #    the model was trained on), comparing with add_generation_prompt=False. This MLX conversion's
    #    template dropped the thinking toggle and ends an open turn at "<|im_start|>assistant\n", so we
    #    deliberately OWN the generation prompt: we append the documented empty think block
    #    ("<think>\n\n</think>\n\n") to force non-thinking, direct answers for RAG. That block is just
    #    input tokens, so it works regardless of the shipped template.
    hf_turns = tokenizer.apply_chat_template(MESSAGES, tokenize=False, add_generation_prompt=False)
    my_turns = hardcoded_render(MESSAGES, add_generation_prompt=False)
    if hf_turns != my_turns:
        print("TURN-FORMAT MISMATCH between HF apply_chat_template and hardcoded_render:", file=sys.stderr)
        print("--- HF ---\n" + repr(hf_turns), file=sys.stderr)
        print("--- mine ---\n" + repr(my_turns), file=sys.stderr)
        sys.exit(1)
    print("turn formatting OK (HF == hardcoded); generation prompt = empty think block (non-thinking)", file=sys.stderr)
    prompt = hardcoded_render(MESSAGES, add_generation_prompt=True)

    # 2. Exact token ids of the rendered prompt.
    token_ids = tokenizer.encode(prompt)
    print(f"prompt tokens: {len(token_ids)}", file=sys.stderr)

    def last_logits(ids):
        out = model(mx.array([ids]))
        return np.array(out[0, -1].astype(mx.float32))

    # 3. Prefill logits (fp32, last position).
    prefill_logits = last_logits(token_ids)

    # 4. Greedy continuation (argmax), 32 steps, recording the top1-top2 gap each step.
    ids = list(token_ids)
    greedy_tokens = []
    greedy_top2_gap = []
    for _ in range(32):
        l = last_logits(ids)
        order = np.argsort(l)[::-1]
        nxt = int(order[0])
        greedy_tokens.append(nxt)
        greedy_top2_gap.append(float(l[order[0]] - l[order[1]]))
        ids.append(nxt)

    out = {
        "model": "Qwen/Qwen3-1.7B-MLX-4bit",
        "messages": MESSAGES,
        "prompt": prompt,
        "token_ids": [int(t) for t in token_ids],
        "prefill_logits": [float(x) for x in prefill_logits.tolist()],
        "greedy_tokens": greedy_tokens,
        "greedy_top2_gap": greedy_top2_gap,
    }
    dst = os.path.join(os.path.dirname(__file__), "..", "Fixtures", "chat_fixtures.json")
    dst = os.path.abspath(dst)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "w") as f:
        json.dump(out, f)
    print(f"wrote {dst} ({len(out['prefill_logits'])} logits, {len(greedy_tokens)} greedy tokens)", file=sys.stderr)


if __name__ == "__main__":
    main()
