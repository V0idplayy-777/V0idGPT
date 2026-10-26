#!/usr/bin/env python3
"""Measure held-out perplexity of a trained checkpoint and of its quantised exports.

Writes models/v0idgpt-4-135M/training-metrics.json so the model card can quote real numbers
instead of guesses. Runs three passes over the same held-out assistant tokens:

  fp32      the checkpoint as trained
  q8emb     int4 everywhere except int8 embeddings  (the shipped export)
  q4all     int4 everywhere

Usage:
  python3 scripts/eval_quant.py --ckpt /tmp/vg/ckpt/final.pt \
      --data models/v0idgpt-4-135M/dataset.jsonl \
      --out models/v0idgpt-4-135M/training-metrics.json
"""
import argparse
import json
import math
import os
import random
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v0id_model import Cfg as ModelConfig, V0idGPT  # noqa: E402


GROUP = 64
IM_START, IM_END = "<|im_start|>", "<|im_end|>"


def quantise_dequantise(t):
    """int4 group-64 symmetric quantisation, the same scheme the browser kernels use."""
    rows, cols = t.shape
    pad = (-cols) % GROUP
    x = t
    if pad:
        x = torch.cat([x, x.new_zeros(rows, pad)], dim=1)
    g = x.view(rows, -1, GROUP)
    scale = g.abs().amax(dim=2, keepdim=True) / 7.0
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    q = torch.clamp(torch.round(g / scale), -8, 7)
    out = (q * scale).view(rows, -1)
    return out[:, :cols].contiguous()


def q8_dequantise(t):
    rows, cols = t.shape
    scale = t.abs().amax(dim=1, keepdim=True) / 127.0
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    q = torch.clamp(torch.round(t / scale), -127, 127)
    return (q * scale).contiguous()


def apply_quant(model, embed_bits):
    """Return a copy of model with quantise-dequantise applied (no-op for fp32)."""
    if embed_bits is None:
        return model
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "embed" in name or "lm_head" in name:
                if embed_bits == 8:
                    p.copy_(q8_dequantise(p.data))
                else:
                    p.copy_(quantise_dequantise(p.data))
            elif p.dim() == 2:
                p.copy_(quantise_dequantise(p.data))
    return model


@torch.no_grad()
def held_out_loss(model, tok, convs, max_len=640, limit=None):
    model.eval()
    total_nll, total_tok = 0.0, 0
    if limit:
        convs = convs[:limit]
    for conv in convs:
        ids, labels = encode_conversation(tok, conv, max_len)
        if not ids:
            continue
        x = torch.tensor([ids[:-1]], dtype=torch.long)
        y = torch.tensor([labels[1:]], dtype=torch.long)
        logits = model(x)
        nll = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1),
                              ignore_index=-100, reduction="sum")
        n = int((y != -100).sum())
        total_nll += float(nll)
        total_tok += n
    return math.exp(total_nll / max(total_tok, 1)), total_tok


def encode_conversation(tok, conv, max_len):
    """Byte-identical to train_sft.render(): ChatML, loss on assistant content + <|im_end|>."""
    ids, mask = [], []

    def add(text, learn):
        t = tok.encode(text, add_special_tokens=False).ids
        ids.extend(t)
        mask.extend([1.0 if learn else 0.0] * len(t))

    sys_prompt = conv.get("system", "")
    if sys_prompt:
        add(f"{IM_START}system\n{sys_prompt}{IM_END}\n", False)
    for m in conv["messages"]:
        if m["role"] == "user":
            add(f"{IM_START}user\n{m['content']}{IM_END}\n", False)
        else:
            add(f"{IM_START}assistant\n", False)
            add(f"{m['content']}{IM_END}\n", True)
    ids, mask = ids[:max_len], mask[:max_len]
    labels = [tok_id if mk else -100 for tok_id, mk in zip(ids, mask)]
    return ids, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="models/v0idgpt-4-135M/dataset.jsonl")
    ap.add_argument("--hf-dir", default="/tmp/vg/hfmodel")
    ap.add_argument("--out", default="models/v0idgpt-4-135M/training-metrics.json")
    ap.add_argument("--holdout", type=int, default=40)
    ap.add_argument("--seed", type=int, default=20251025)
    args = ap.parse_args()

    blob = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ModelConfig(**blob["cfg"])
    model = V0idGPT(cfg)
    model.load_state_dict(blob["state_dict"])
    n_params = sum(p.numel() for p in model.parameters())

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(args.hf_dir, "tokenizer.json"))

    convs = [json.loads(line) for line in open(args.data) if line.strip()]
    rng = random.Random(args.seed)
    rng.shuffle(convs)
    held = convs[: args.holdout]

    result = {
        "model": "v0idgpt-4-135M",
        "params": n_params,
        "checkpoint": os.path.basename(args.ckpt),
        "step": blob.get("step"),
        "holdout_conversations": len(held),
        "perplexity": {},
    }
    for label, bits in (("fp32", None), ("q8_embeddings", 8), ("q4_all", 4)):
        m = V0idGPT(cfg)
        m.load_state_dict(blob["state_dict"])
        apply_quant(m, bits)
        ppl, ntok = held_out_loss(m, tok, held)
        result["perplexity"][label] = {"ppl": round(ppl, 4), "tokens": ntok}
        print(label, round(ppl, 4), ntok, flush=True)
        del m

    base = result["perplexity"]["fp32"]["ppl"]
    for label in ("q8_embeddings", "q4_all"):
        v = result["perplexity"][label]["ppl"]
        result["perplexity"][label]["delta_vs_fp32_pct"] = round((v / base - 1) * 100, 2)

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
