#!/usr/bin/env python3
"""
Fine-tune v0idgpt-4-135M on the hand-written + generated conversational corpus.

  python3 scripts/train_sft.py --base /path/to/base --out /tmp/vg/ckpt --epochs 3

The base weights are SmolLM2-135M-Instruct (bf16 safetensors, or the chunked staging layout this
repo's scripts/README describes). Only the assistant tokens carry loss; system and user tokens are
masked out, which is what teaches "reply to the last message" rather than "continue the transcript".

Training notes:
  * fp32 CPU training, gradient checkpointing, batch 2 x accum 2, bucketed lengths (little padding)
  * AdamW, warmup + cosine decay, grad-clip 1.0
  * checkpoint + resume every N steps, held-out loss and greedy samples written to the log
"""
from __future__ import annotations

import argparse, json, math, os, random, sys, time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v0id_model import Cfg, V0idGPT, load_chunked, load_safetensors, hf_names_to_ours, build_model

IM_START, IM_END = "<|im_start|>", "<|im_end|>"

PROBES = [
    "hi",
    "nothing",
    "what is 6 times 8",
    "what's 47 + 38",
    "no",
    "what's your name?",
    "i'm feeling sad today",
    "how many parameters do you have",
]


# --------------------------------------------------------------------------- data
def render(conv, tok, max_len):
    """ChatML transcript -> (ids, loss_mask) with loss only on assistant content + <|im_end|>."""
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
    if len(ids) > max_len:
        ids, mask = ids[:max_len], mask[:max_len]
    return ids, mask


class BucketedBatcher:
    """Length-sorted batches with minimal padding, reshuffled each epoch."""

    def __init__(self, samples, batch_size, seed):
        self.samples = sorted(samples, key=lambda s: len(s[0]))
        self.bs = batch_size
        self.rng = random.Random(seed)

    def __iter__(self):
        idx = list(range(len(self.samples)))
        # shuffle inside strata so batches stay length-homogeneous but order varies
        stride = self.bs * 25
        for i in range(0, len(idx), stride):
            block = idx[i:i + stride]
            self.rng.shuffle(block)
            idx[i:i + stride] = block
        for i in range(0, len(idx), self.bs):
            group = [self.samples[j] for j in idx[i:i + self.bs]]
            pad = max(len(g[0]) for g in group)
            ids = torch.zeros(len(group), pad, dtype=torch.long)
            mask = torch.zeros(len(group), pad)
            attn = torch.zeros(len(group), pad, dtype=torch.bool)
            for r, (seq, lm) in enumerate(group):
                ids[r, :len(seq)] = torch.tensor(seq)
                mask[r, :len(lm)] = torch.tensor(lm)
                attn[r, :len(seq)] = True
            yield ids, mask, attn


def build_samples(tok, path, max_len):
    out = []
    for line in open(path, encoding="utf-8"):
        conv = json.loads(line)
        ids, mask = render(conv, tok, max_len)
        if sum(mask) < 4:            # nothing to learn from
            continue
        out.append((ids, mask))
    return out


# --------------------------------------------------------------------------- generation
@torch.no_grad()
def greedy(model, tok, prompt_ids, max_new=40, stop_at_eos=True):
    model.eval()
    ids = list(prompt_ids)
    eos = tok.token_to_id(IM_END)
    for _ in range(max_new):
        x = torch.tensor([ids[-model.cfg.max_position_embeddings:]])
        logits = model(x)
        nxt = int(logits[0, -1].argmax())
        ids.append(nxt)
        if stop_at_eos and nxt == eos:
            break
    return ids[len(prompt_ids):]


def probe_text(tok, q, system):
    s = f"{IM_START}system\n{system}{IM_END}\n{IM_START}user\n{q}{IM_END}\n{IM_START}assistant\n"
    return tok.encode(s, add_special_tokens=False).ids


def run_probes(model, tok, system, n=6):
    out = []
    for q in PROBES[:n]:
        ids = probe_text(tok, q, system)
        gen = greedy(model, tok, ids, max_new=48)
        out.append((q, tok.decode(gen, skip_special_tokens=True).strip()))
    model.train()
    return out


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="/tmp/vg/base")
    ap.add_argument("--base-safetensors", default=None)
    ap.add_argument("--hf-dir", default="/tmp/vg/hfmodel", help="dir with tokenizer.json")
    ap.add_argument("--data", default="models/v0idgpt-4-135M/dataset.jsonl")
    ap.add_argument("--out", default="/tmp/vg/ckpt")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--max-len", type=int, default=640)
    ap.add_argument("--lr", type=float, default=2.5e-4)
    ap.add_argument("--min-lr-ratio", type=float, default=0.08)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--val-frac", type=float, default=0.02)
    ap.add_argument("--log-every", type=int, default=5)
    ap.add_argument("--probe-every", type=int, default=40)
    ap.add_argument("--save-every", type=int, default=100)
    ap.add_argument("--smoke", action="store_true", help="a few steps only, for pipeline checks")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--fresh-optimizer", action="store_true",
                    help="resume weights only: new AdamW state, step counter back to 0 (new training stage)")
    args = ap.parse_args()

    torch.set_num_threads(2)
    torch.manual_seed(1337)
    random.seed(1337)
    os.makedirs(args.out, exist_ok=True)
    log_path = os.path.join(args.out, "train.log")
    logf = open(log_path, "a", encoding="utf-8")

    def cur_rss_mb():
        try:
            with open("/proc/self/statm") as f:
                return round(int(f.read().split()[1]) * 4096 / 1e6, 1)
        except Exception:
            return None

    def log(rec):
        if isinstance(rec, dict) and "step" in rec:
            rec["rss_mb"] = cur_rss_mb()
        line = json.dumps(rec) if isinstance(rec, dict) else str(rec)
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(args.hf_dir, "tokenizer.json"))

    # ---- model
    if args.resume:
        blob = torch.load(args.resume, map_location="cpu", weights_only=False)
        cfg = Cfg(**blob["cfg"])
        model = V0idGPT(cfg)
        model.load_state_dict(blob["state_dict"])
        log({"resumed": args.resume, "step": blob.get("step")})
    else:
        if args.base_safetensors:
            sd = hf_names_to_ours(load_safetensors(args.base_safetensors))
            cfg = Cfg.from_json(os.path.join(args.hf_dir, "config.json"))
        else:
            sd, cfg = load_chunked(args.base)
            sd = hf_names_to_ours(sd)
        model = build_model(sd, cfg)
        del sd
    log({"params": model.num_params(), "tied": model.cfg.tie_word_embeddings,
         "layers": cfg.num_hidden_layers, "hidden": cfg.hidden_size, "vocab": cfg.vocab_size})
    model.train()
    model.config = cfg  # handy for generation helpers

    # ---- gradient checkpointing (memory is the binding constraint on CPU)
    blocks = model.layers
    def enable_ckpt(flag=True):
        for blk in blocks:
            orig = blk.forward
            if flag:
                blk.forward = lambda x, cos, sin, mask=None, _o=orig: torch.utils.checkpoint.checkpoint(
                    _o, x, cos, sin, mask, use_reentrant=False)
    enable_ckpt(True)

    # ---- data
    samples = build_samples(tok, args.data, args.max_len)
    rng = random.Random(7)
    rng.shuffle(samples)
    nval = max(8, int(len(samples) * args.val_frac))
    val, train = samples[:nval], samples[nval:]
    log({"train_convs": len(train), "val_convs": len(val),
         "train_assistant_tokens": int(sum(sum(m) for _, m in train))})

    steps_per_epoch = math.ceil(len(train) / (args.batch_size * args.accum))
    total_steps = max(1, int(steps_per_epoch * args.epochs))
    if args.smoke:
        total_steps = 12
    log({"steps_per_epoch": steps_per_epoch, "total_steps": total_steps, "lr": args.lr})

    decay, no_decay = [], []
    for p in model.parameters():
        (decay if p.dim() >= 2 else no_decay).append(p)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.wd},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, 0.95), eps=1e-8, foreach=False)  # foreach peak-allocates per group; plain loop is gentler on a small box
    start_step = 0
    if args.resume and not args.fresh_optimizer:
        st = torch.load(args.resume, map_location="cpu", weights_only=False)
        if "optimizer" in st and not isinstance(st["optimizer"], str):
            opt.load_state_dict(st["optimizer"])
            start_step = st.get("step", 0)

    # glibc fragments badly under a long sequence of large CPU allocations; hand pages back
    # regularly or the process creeps to the container limit and gets OOM-killed mid-run.
    try:
        import ctypes
        _libc = ctypes.CDLL("libc.so.6")
        trim = lambda: _libc.malloc_trim(0)
    except Exception:
        trim = lambda: None

    def lr_at(step):
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        prog = (step - args.warmup) / max(1, total_steps - args.warmup)
        prog = min(1.0, prog)
        cos = 0.5 * (1 + math.cos(math.pi * prog))
        return args.lr * (args.min_lr_ratio + (1 - args.min_lr_ratio) * cos)

    # ---- validation set as one fixed batch
    def eval_loss():
        model.eval()
        tot, n = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(val), 4):
                group = val[i:i + 4]
                pad = max(len(g[0]) for g in group)
                ids = torch.zeros(len(group), pad, dtype=torch.long)
                mask = torch.zeros(len(group), pad)
                attn = torch.zeros(len(group), pad, dtype=torch.bool)
                for r, (s, lm) in enumerate(group):
                    ids[r, :len(s)] = torch.tensor(s)
                    mask[r, :len(lm)] = torch.tensor(lm)
                    attn[r, :len(s)] = True
                inp, tgt, lm = ids[:, :-1], ids[:, 1:], mask[:, 1:]
                logits = model(inp, attention_mask=attn[:, :-1])
                l = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1),
                                    reduction="none").view(tgt.shape)
                tot += float((l * lm).sum()); n += int(lm.sum())
        model.train()
        return tot / max(1, n)

    t0 = time.time()
    step, seen_tokens, run_loss, run_n = start_step, 0, 0.0, 0
    epoch = 0.0
    done = start_step
    while done < total_steps:
        batcher = BucketedBatcher(train, args.batch_size, seed=1300 + int(epoch))
        epoch += 1
        opt.zero_grad(set_to_none=True)
        for bi, (ids, mask, attn) in enumerate(batcher):
            if done >= total_steps:
                break
            inp, tgt, lm = ids[:, :-1], ids[:, 1:], mask[:, 1:]
            logits = model(inp, attention_mask=attn[:, :-1])
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1),
                                   reduction="none").view(tgt.shape)
            loss = (loss * lm).sum() / lm.sum().clamp(min=1.0)
            (loss / args.accum).backward()
            run_loss += float(loss.detach()); run_n += 1
            seen_tokens += int(attn.sum())
            if (bi + 1) % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                for g in opt.param_groups:
                    g["lr"] = lr_at(done)
                opt.step()
                opt.zero_grad(set_to_none=True)
                done += 1
                if done % 10 == 0:
                    trim()
                if done % args.log_every == 0:
                    el = time.time() - t0
                    tps = seen_tokens / max(1e-9, el)
                    log({"step": done, "total": total_steps, "epoch": round(epoch, 2),
                         "loss": round(run_loss / max(1, run_n), 4),
                         "lr": f"{lr_at(done):.2e}", "tok_per_s": round(tps, 1),
                         "elapsed_min": round(el / 60, 1)})
                    run_loss, run_n = 0.0, 0
                if done % args.probe_every == 0 or args.smoke:
                    vl = eval_loss()
                    samples_out = run_probes(model, tok, "You are v0idgpt, a small assistant running locally in the browser. Answer what you're asked, be concise, and admit uncertainty.", 6)
                    log({"step": done, "val_loss": round(vl, 4)})
                    for q, a in samples_out:
                        log(f"   Q: {q}\n   A: {a}")
                if done % args.save_every == 0 or done >= total_steps:
                    p = os.path.join(args.out, f"step{done}.pt")
                    only_weights = args.fresh_optimizer  # optimizer state is unused by the next stage
                    blob = {"cfg": cfg.__dict__, "state_dict": model.state_dict(), "step": done}
                    if not only_weights:
                        blob["optimizer"] = opt.state_dict()
                    else:
                        blob["optimizer"] = "omitted"
                        blob["weights_only"] = True
                    torch.save(blob, p)
                    torch.save({"cfg": cfg.__dict__, "state_dict": model.state_dict(),
                                "step": done}, os.path.join(args.out, "latest.pt"))
                    log({"saved": p})
    log({"done": True, "minutes": round((time.time() - t0) / 60, 1)})
    torch.save({"cfg": cfg.__dict__, "state_dict": model.state_dict(), "step": done},
               os.path.join(args.out, "final.pt"))
    log({"saved_final": os.path.join(args.out, "final.pt")})


if __name__ == "__main__":
    main()
