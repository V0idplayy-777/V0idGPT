#!/usr/bin/env python3
"""Compute reference logits from an exported model.bin, for cross-checking the JavaScript engine.

  python3 scripts/reference_logits.py --bin models/v0idgpt-4-135M/model.bin --out /tmp/ref.json

Dequantises the int4/int8 payload with numpy (exactly the maths the browser does), runs the model in
torch, and writes the logits for a few prompts plus the greedy continuation.
"""
import argparse, json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v0id_model import Cfg, V0idGPT
from export_model import MAGIC, dequant_back

PROMPTS = [
    "<|im_start|>system\nYou are v0idgpt, a small assistant running locally in the browser. Answer what you're asked, be concise, and admit uncertainty.<|im_end|>\n<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n",
    "<|im_start|>user\nwhat is 6 times 8<|im_end|>\n<|im_start|>assistant\n",
    "<|im_start|>user\nnothing<|im_end|>\n<|im_start|>assistant\n",
]


def load_export(path):
    data = open(path, "rb").read()
    assert data[:8] == MAGIC, "bad magic"
    hlen = int.from_bytes(data[8:12], "little")
    header = json.loads(data[12:12 + hlen])
    base = 12 + hlen
    sd = {}
    for t in header["tensors"]:
        spec = {"kind": t["kind"], "shape": t["shape"], "data": data[base + t["offset"]: base + t["offset"] + t["bytes"]]}
        if t["kind"] == "q4":
            spec["group"] = t["group"]
            spec["scales"] = data[base + t["scales"]: base + t["scales"] + t["scale_bytes"]]
            spec["scale_shape"] = t["scale_shape"]
        elif t["kind"] == "q8":
            spec["scales"] = data[base + t["scales"]: base + t["scales"] + t["scale_bytes"]]
            spec["scale_shape"] = t["scale_shape"]
        sd[t["name"]] = dequant_back(spec)
    return header, sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hf-dir", default="/tmp/vg/hfmodel")
    ap.add_argument("--continue-tokens", type=int, default=6)
    args = ap.parse_args()
    torch.set_num_threads(2)

    header, sd = load_export(args.bin)
    cfg = Cfg(**{k: v for k, v in header["config"].items() if k in Cfg.__dataclass_fields__})
    model = V0idGPT(cfg)
    model.load_state_dict(sd, strict=False)
    model.eval()
    print("loaded", args.bin, "params", model.num_params())

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(args.hf_dir, "tokenizer.json"))

    out = []
    for prompt in PROMPTS:
        ids = tok.encode(prompt, add_special_tokens=False).ids
        with torch.no_grad():
            logits = model(torch.tensor([ids]))[0, -1]
        top = torch.topk(logits, 10)
        cont = list(ids)
        cont_tokens = []
        for _ in range(args.continue_tokens):
            with torch.no_grad():
                lg = model(torch.tensor([cont]))[0, -1]
            nxt = int(lg.argmax())
            cont.append(nxt)
            cont_tokens.append(nxt)
        out.append({
            "prompt": prompt, "ids": ids,
            "top10": [[int(i), float(v)] for i, v in zip(top.indices, top.values)],
            "last_logits_absmax": float(logits.abs().max()),
            "greedy": cont_tokens,
            "greedy_text": tok.decode(cont_tokens),
        })
        print(f"  prompt[{len(ids)} tok] greedy -> {out[-1]['greedy_text']!r}")
    json.dump({"model": args.bin, "cases": out}, open(args.out, "w"), indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
