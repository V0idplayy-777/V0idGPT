"""
v0idGPT model definition (Llama architecture) — plain PyTorch, no transformers dependency.

Used by:
  scripts/train_sft.py     training
  scripts/verify_model.py  numerical checks
  scripts/export_web.py    conversion to the quantised browser format

RoPE is the non-interleaved (rotate-half) variant, RMSNorm epsilon 1e-5, SwiGLU MLP, GQA attention,
and the embedding matrix is tied to the output head — all matching SmolLM2-135M's config.
"""
from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Cfg:
    vocab_size: int = 49152
    hidden_size: int = 576
    intermediate_size: int = 1536
    num_hidden_layers: int = 30
    num_attention_heads: int = 9
    num_key_value_heads: int = 3
    max_position_embeddings: int = 8192
    rms_norm_eps: float = 1e-5
    rope_theta: float = 100000.0
    tie_word_embeddings: bool = True

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @classmethod
    def from_json(cls, path):
        d = json.load(open(path))
        keys = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in keys})


def rope_tables(head_dim, seq_len, theta, device="cpu", dtype=torch.float32):
    """cos/sin of shape [seq_len, head_dim] for the rotate-half convention."""
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv)                      # [T, head_dim/2]
    emb = torch.cat((freqs, freqs), dim=-1)          # [T, head_dim]
    return emb.cos().to(dtype), emb.sin().to(dtype)


def rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        d = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x.to(d) * self.weight


class Attention(nn.Module):
    def __init__(self, cfg: Cfg):
        super().__init__()
        self.nh, self.nkv, self.hd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        self.q_proj = nn.Linear(cfg.hidden_size, self.nh * self.hd, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, self.nkv * self.hd, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, self.nkv * self.hd, bias=False)
        self.o_proj = nn.Linear(self.nh * self.hd, cfg.hidden_size, bias=False)

    def forward(self, x, cos, sin, mask=None):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.nh, self.hd).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.nkv, self.hd).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.nkv, self.hd).transpose(1, 2)
        c, s = cos[:T].to(q.dtype), sin[:T].to(q.dtype)
        q = q * c + rotate_half(q) * s
        k = k * c + rotate_half(k) * s
        if self.nkv != self.nh:                       # repeat KV heads for GQA
            rep = self.nh // self.nkv
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        # scaled dot-product attention; bool mask True = may attend
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.hd)
        causal = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
        if mask is not None:
            causal = causal & mask[:, None, None, :]
        att = att.masked_fill(~causal, float("-inf")).softmax(-1)
        out = (att @ v).transpose(1, 2).contiguous().view(B, T, self.nh * self.hd)
        return self.o_proj(out)


class MLP(nn.Module):
    def __init__(self, cfg: Cfg):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, cfg: Cfg):
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.self_attn = Attention(cfg)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mlp = MLP(cfg)

    def forward(self, x, cos, sin, mask=None):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, mask)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class V0idGPT(nn.Module):
    def __init__(self, cfg: Cfg):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList([Block(cfg) for _ in range(cfg.num_hidden_layers)])
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    def num_params(self) -> int:
        seen, total = set(), 0
        for p in self.parameters():
            if id(p) in seen:
                continue
            seen.add(id(p))
            total += p.numel()
        return total

    def forward(self, idx, targets=None, attention_mask=None, loss_mask=None):
        B, T = idx.shape
        cos, sin = rope_tables(self.cfg.head_dim, T, self.cfg.rope_theta, idx.device, torch.float32)
        x = self.embed_tokens(idx)
        for blk in self.layers:
            x = blk(x, cos, sin, attention_mask)
        logits = self.lm_head(self.norm(x))
        if targets is None:
            return logits
        # cast only the positions we score: avoids a [B,T,V] float32 tensor when masking
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)).float(), targets.reshape(-1), reduction="none"
        ).view(B, T)
        if loss_mask is not None:
            loss = loss * loss_mask
            denom = loss_mask.sum().clamp(min=1.0)
            return (loss.sum() / denom), logits
        return loss.mean(), logits


# --------------------------------------------------------------------------- weight loading
DT = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32, "BF16 ": torch.bfloat16}


def load_chunked(base_dir, dtype=torch.float32):
    """Load the chunked staging layout: manifest.json + chunks/w_000.bin ..."""
    import numpy as np
    man = json.load(open(f"{base_dir}/manifest.json"))
    bufs = {}
    sd = {}
    for t in man["tensors"]:
        i = t["chunk"]
        if i not in bufs:
            bufs[i] = open(f"{base_dir}/chunks/w_{i:03d}.bin", "rb").read()
        raw = bufs[i][t["offset"]: t["offset"] + t["nbytes"]]
        x = torch.frombuffer(bytearray(raw), dtype=DT[t["dtype"]]).reshape(t["shape"])
        sd[t["name"]] = x.to(dtype)
    return sd, Cfg(**{k: v for k, v in man["config"].items() if k in Cfg.__dataclass_fields__})


def load_safetensors(path, dtype=torch.float32):
    from safetensors.torch import load_file
    return {k: v.to(dtype) for k, v in load_file(path).items()}


def build_model(sd, cfg, dtype=torch.float32):
    m = V0idGPT(cfg).to(dtype)
    missing, unexpected = m.load_state_dict(sd, strict=False)
    if any("layers" in k for k in missing):
        raise RuntimeError(f"missing weights: {missing[:5]}")
    return m


def hf_names_to_ours(sd):
    """HF LlamaForCausalLM state dict -> our module's names."""
    out = {}
    for k, v in sd.items():
        k2 = k
        if k.startswith("model."):
            k2 = k[len("model."):]
        out[k2] = v
    return out


def save_ours(m, path):
    torch.save({"cfg": m.cfg.__dict__, "state_dict": m.state_dict()}, path)


def load_ours(path, dtype=torch.float32):
    blob = torch.load(path, map_location="cpu", weights_only=False)
    m = V0idGPT(Cfg(**blob["cfg"])).to(dtype)
    m.load_state_dict(blob["state_dict"])
    return m


if __name__ == "__main__":
    import sys
    torch.set_num_threads(2)
    base = sys.argv[1] if len(sys.argv) > 1 else "/tmp/vg/base"
    sd, cfg = load_chunked(base)
    sd = hf_names_to_ours(sd)
    m = build_model(sd, cfg)
    print("config:", cfg)
    print("params:", f"{m.num_params():,}")
