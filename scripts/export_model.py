#!/usr/bin/env python3
"""
Export a trained v0idgpt checkpoint into the compact browser format.

  python3 scripts/export_model.py --ckpt /tmp/vg/ckpt/final.pt --out models/v0idgpt-4-135M/model.bin

Format (single file, little endian):
  magic  b"V0IDGPT4"  |  u32 header_len  |  header JSON  |  payload
  header = {format, config, tokens, embed, layers:[...], tensors:[{name,kind,shape,...}]}

Quantisation
  q4  : signed 4-bit, symmetric, groups of 64 along the input dim, one fp16 scale per group.
        Two nibbles per byte; low nibble = even index. 4.25 bits/parameter.
  q8  : signed 8-bit, symmetric, one fp16 scale per row (used for embeddings when --embed-bits 8).
  f16 : raw fp16, for the RMSNorm vectors.

The dequantisation the browser does is exactly  w = q * scale[group].  Nothing else is stored.
"""
from __future__ import annotations

import argparse, json, os, struct, sys, time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v0id_model import Cfg, V0idGPT

MAGIC = b"V0IDGPT4"
GROUP = 64


# ------------------------------------------------------------------ quantisation primitives
def q4_pack(w: np.ndarray, group=GROUP):
    """w: [rows, cols] float32 -> (packed uint8 [rows, ceil(cols/2)], scales fp16 [rows, ceil(cols/group)])"""
    rows, cols = w.shape
    ngroups = (cols + group - 1) // group
    pad = ngroups * group - cols
    if pad:
        w = np.pad(w, ((0, 0), (0, pad)))
    wr = w.reshape(rows, ngroups, group)
    scale = np.abs(wr).max(axis=2) / 7.0
    scale[scale == 0] = 1e-8
    q = np.rint(wr / scale[..., None]).clip(-8, 7).astype(np.int8)
    flat = q.reshape(rows, ngroups * group)
    lo, hi = flat[:, 0::2], flat[:, 1::2]
    packed = ((lo & 0x0F) | ((hi & 0x0F) << 4)).astype(np.uint8)
    return packed, scale.astype(np.float16)


def q4_unpack(packed: np.ndarray, scale: np.ndarray, rows, cols, group=GROUP):
    lo = (packed & 0x0F).astype(np.int8)
    hi = ((packed >> 4) & 0x0F).astype(np.int8)
    lo = np.where(lo > 7, lo - 16, lo)
    hi = np.where(hi > 7, hi - 16, hi)
    flat = np.empty((rows, packed.shape[1] * 2), dtype=np.int8)
    flat[:, 0::2] = lo
    flat[:, 1::2] = hi
    flat = flat[:, :cols].reshape(rows, -1, group)
    out = flat.astype(np.float32) * scale.astype(np.float32)[..., None]
    return out.reshape(rows, -1)[:, :cols]


def q8_pack(w: np.ndarray):
    scale = np.abs(w).max(axis=1) / 127.0
    scale[scale == 0] = 1e-8
    q = np.rint(w / scale[:, None]).clip(-127, 127).astype(np.int8)
    return q, scale.astype(np.float16)


def q8_unpack(q: np.ndarray, scale: np.ndarray):
    return q.astype(np.float32) * scale.astype(np.float32)[:, None]


def quantize_tensor(name, w: torch.Tensor, embed_bits: int):
    a = w.detach().to(torch.float32).numpy()
    is_2d = a.ndim == 2
    if not is_2d:
        return {"name": name, "kind": "f16", "shape": list(a.shape), "data": a.astype(np.float16).tobytes()}
    rows, cols = a.shape
    if embed_bits == 8 and ("embed_tokens" in name or "lm_head" in name):
        q, s = q8_pack(a)
        return {"name": name, "kind": "q8", "shape": [rows, cols],
                "data": q.tobytes(), "scales": s.tobytes(), "scale_shape": [rows]}
    q, s = q4_pack(a)
    return {"name": name, "kind": "q4", "shape": [rows, cols], "group": GROUP,
            "data": q.tobytes(), "scales": s.tobytes(), "scale_shape": list(s.shape)}


def dequant_back(spec):
    kind = spec["kind"]
    if kind == "f16":
        return torch.from_numpy(np.frombuffer(spec["data"], dtype=np.float16).copy()).to(torch.float32)
    rows, cols = spec["shape"]
    if kind == "q8":
        q = np.frombuffer(spec["data"], dtype=np.int8).reshape(rows, cols)
        s = np.frombuffer(spec["scales"], dtype=np.float16).reshape(spec["scale_shape"])
        return torch.from_numpy(q8_unpack(q, s))
    packed = np.frombuffer(spec["data"], dtype=np.uint8).reshape(rows, -1)
    scales = np.frombuffer(spec["scales"], dtype=np.float16).reshape(spec["scale_shape"])
    return torch.from_numpy(q4_unpack(packed, scales, rows, cols, spec["group"]))


# ------------------------------------------------------------------ perplexity check
@torch.no_grad()
def perplexity(model, tok, data_path, n_convs=120, max_len=384):
    """Assistant-token perplexity on held-out conversations (lower is better)."""
    from train_sft import render
    model.eval()
    total_nll, total_tok = 0.0, 0
    lines = open(data_path, encoding="utf-8").read().splitlines()
    step = max(1, len(lines) // n_convs)
    picked = lines[::step][:n_convs]
    for line in picked:
        conv = json.loads(line)
        ids, mask = render(conv, tok, max_len)
        if sum(mask) < 4:
            continue
        x = torch.tensor([ids[:-1]])
        tgt = torch.tensor([ids[1:]])
        lm = torch.tensor([mask[1:]])
        logits = model(x)
        nll = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), tgt.view(-1), reduction="none").view(tgt.shape)
        total_nll += float((nll * lm).sum())
        total_tok += int(lm.sum())
    return float(torch.exp(torch.tensor(total_nll / max(1, total_tok))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--embed-bits", type=int, default=4, choices=[4, 8])
    ap.add_argument("--eval", action="store_true", help="compare fp32 vs quantised perplexity")
    ap.add_argument("--data", default="models/v0idgpt-4-135M/dataset.jsonl")
    ap.add_argument("--hf-dir", default="/tmp/vg/hfmodel")
    args = ap.parse_args()

    torch.set_num_threads(2)
    blob = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = Cfg(**blob["cfg"])
    model = V0idGPT(cfg)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    print(f"checkpoint: {args.ckpt}  step={blob.get('step')}  params={model.num_params():,}")

    if args.eval:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(os.path.join(args.hf_dir, "tokenizer.json"))
        ppl_fp32 = perplexity(model, tok, args.data)
        print(f"  fp32            ppl {ppl_fp32:.3f}")
        for bits in (4, 8):
            specs = [quantize_tensor(n, p, bits) for n, p in model.named_parameters()]
            sd = {s["name"]: dequant_back(s) for s in specs}
            q = V0idGPT(cfg)
            q.load_state_dict(sd, strict=False)
            ppl = perplexity(q, tok, args.data)
            size = sum(len(s["data"]) + len(s.get("scales", b"")) for s in specs)
            print(f"  embed_bits={bits}  ppl {ppl:.3f}  (+{(ppl/ppl_fp32-1)*100:.2f}%)  {size/1e6:.1f} MB")
        return

    if not args.out:
        raise SystemExit("--out required unless --eval")

    specs = []
    for name, p in model.named_parameters():
        specs.append(quantize_tensor(name, p, args.embed_bits))
    header = {
        "format": "v0idgpt-q4-v1",
        "arch": "llama",
        "name": "v0idgpt-4-135M",
        "source": "SmolLM2-135M-Instruct + v0idgpt-4 conversational SFT",
        "config": cfg.__dict__,
        "params": model.num_params(),
        "embed_bits": args.embed_bits,
        "group": GROUP,
        "quant": "q4 (symmetric, group 64, fp16 scales) + q8 embeddings" if args.embed_bits == 8
                 else "q4 (symmetric, group 64, fp16 scales) for all matrices",
        "tensors": [],
    }
    payload = bytearray()

    def align4():
        while len(payload) % 4:
            payload.append(0)

    for s in specs:
        align4()
        off = len(payload)
        payload += s["data"]
        entry = {"name": s["name"], "kind": s["kind"], "shape": s["shape"],
                 "offset": off, "bytes": len(s["data"])}
        if s["kind"] == "q4":
            entry["group"] = s["group"]
            align4()
            entry["scales"] = len(payload)
            entry["scale_shape"] = s["scale_shape"]
            payload += s["scales"]
            entry["scale_bytes"] = len(s["scales"])
        elif s["kind"] == "q8":
            align4()
            entry["scales"] = len(payload)
            entry["scale_shape"] = s["scale_shape"]
            payload += s["scales"]
            entry["scale_bytes"] = len(s["scales"])
        header["tensors"].append(entry)
    hjson = json.dumps(header, separators=(",", ":")).encode()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<I", len(hjson)))
        f.write(hjson)
        f.write(payload)
    mb = os.path.getsize(args.out) / 1e6
    print(f"wrote {args.out}  {mb:.1f} MB  ({len(specs)} tensors, header {len(hjson)/1024:.1f} KB)")
    print(json.dumps({k: header[k] for k in ("format", "params", "embed_bits", "quant")}, indent=1))


if __name__ == "__main__":
    main()
