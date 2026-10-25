#!/usr/bin/env python3
"""Train and export v0idgpt-3-3M-reasoning in NeuronForge's format.

Only assistant continuations contribute to the loss: the user's prompt is
context, not something the model is asked to imitate. The browser runtime stays
zero-dependency and generates text from the exported float32 weights.
"""
from __future__ import annotations

import argparse
import base64
import json
import random
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models/v0idgpt-3-3M-reasoning"
DATA = MODEL_DIR / "dataset.jsonl"
SEQ, EMBED, HIDDEN = 128, 128, 800
# Runtime character tokenization lowercases input. ID 0 is left-padding.
TOKENS = ["<pad>", "\n"] + list(" !\"#$%&'()*+,-./0123456789:;<=>?@[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~")
VOCAB = len(TOKENS)
STOI = {c: i for i, c in enumerate(TOKENS)}


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, EMBED, padding_idx=0)
        self.lstm = nn.LSTM(EMBED, HIDDEN, batch_first=True)
        self.logits = nn.Linear(HIDDEN, VOCAB)

    def forward(self, x):
        y, _ = self.lstm(self.embed(x))
        return self.logits(y)


def encode(text: str) -> list[int]:
    return [STOI.get(ch, STOI[" "]) for ch in text.lower()]


def load_examples() -> list[dict]:
    examples = []
    with DATA.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            row = json.loads(line)
            parts = []
            for turn in row.get("history", []):
                parts.append(f"user: {turn['user']}\nassistant: {turn['assistant']}\n")
            parts.append(f"user: {row['user']}\nassistant:")
            prefix = encode("".join(parts))
            # Browser inference only keeps the last SEQ characters. Keep the
            # training prefix bounded the same way, while preserving the most
            # recent question and its assistant marker.
            prefix = prefix[-SEQ:]
            answer = encode(" " + row["assistant"] + "\n")
            full = prefix + answer
            if len(full) < 2 or not answer:
                raise ValueError(f"Empty example at {DATA}:{line_no}")
            x = full[:-1]
            y = full[1:]
            # x[t] predicts y[t]; the first assistant character is at
            # full[len(prefix)], so its target position is len(prefix) - 1.
            first_answer_target = max(0, len(prefix) - 1)
            weight = float(row.get("weight", 1.0))
            if not weight > 0:
                raise ValueError(f"Invalid example weight at {DATA}:{line_no}")
            mask = [weight if i >= first_answer_target else 0.0 for i in range(len(x))]
            examples.append({"x": x, "y": y, "mask": mask, "category": row.get("category", "unknown")})
    if not examples:
        raise ValueError(f"No training examples found in {DATA}")
    return examples


def make_batch(rows: list[dict], device: torch.device):
    batch = len(rows)
    length = max(len(row["x"]) for row in rows)
    x = torch.zeros((batch, length), dtype=torch.long, device=device)
    y = torch.zeros((batch, length), dtype=torch.long, device=device)
    mask = torch.zeros((batch, length), dtype=torch.float32, device=device)
    for i, row in enumerate(rows):
        n = len(row["x"])
        x[i, :n] = torch.tensor(row["x"], dtype=torch.long, device=device)
        y[i, :n] = torch.tensor(row["y"], dtype=torch.long, device=device)
        mask[i, :n] = torch.tensor(row["mask"], dtype=torch.float32, device=device)
    return x, y, mask


def loss_for(model: Model, batch: list[dict], device: torch.device):
    x, y, mask = make_batch(batch, device)
    logits = model(x)
    per_character = F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1), reduction="none").reshape_as(mask)
    count = mask.sum()
    loss_sum = (per_character * mask).sum()
    return loss_sum, count


def checksum(key: str, raw: bytes) -> str:
    all_bytes = key.encode() + raw
    s1 = sum(all_bytes) & 0xffffffff
    s2 = sum((i % 65521 + 1) * b for i, b in enumerate(all_bytes)) & 0xffffffff
    return f"{s1:08x}{s2:08x}"


def tensor_entry(key: str, tensor: torch.Tensor) -> dict:
    tensor = tensor.detach().cpu().contiguous().float()
    raw = tensor.numpy().tobytes()
    return {
        "shape": list(tensor.shape),
        "dtype": "float32",
        "data": base64.b64encode(raw).decode(),
        "checksum": checksum(key, raw),
    }


def export(model: Model, metrics: dict) -> None:
    model.eval()
    reference_prompt = "user: hello\nassistant:"
    reference_ids = encode(reference_prompt)
    reference_input = [0] * (SEQ - len(reference_ids)) + reference_ids
    with torch.no_grad():
        reference_logits = model(torch.tensor([reference_input], dtype=torch.long))[0, -1]
        reference_output = torch.softmax(reference_logits, dim=0).tolist()
    embedding_id, lstm_id, dense_id = "embedding_3m", "lstm_3m", "logits_3m"
    # PyTorch and NeuronForge both use i,f,g,o gates. Runtime matrices are [in,out].
    tensors = {
        f"{embedding_id}.W": model.embed.weight,
        f"{lstm_id}.Wx": model.lstm.weight_ih_l0.T,
        f"{lstm_id}.Wh": model.lstm.weight_hh_l0.T,
        f"{lstm_id}.b": model.lstm.bias_ih_l0 + model.lstm.bias_hh_l0,
        f"{dense_id}.W": model.logits.weight.T,
        f"{dense_id}.b": model.logits.bias,
    }
    weights = {key: tensor_entry(key, value) for key, value in tensors.items()}
    param_count = sum(value.numel() for value in tensors.values())
    layers = [
        {"id": embedding_id, "name": "character embedding", "type": "embedding", "config": {"vocabSize": VOCAB, "embeddingDim": EMBED}, "inputShape": [SEQ], "outputShape": [SEQ, EMBED]},
        {"id": lstm_id, "name": "conversational memory", "type": "lstm", "config": {"units": HIDDEN, "returnSequences": False}, "inputShape": [SEQ, EMBED], "outputShape": [HIDDEN]},
        {"id": dense_id, "name": "next character logits", "type": "dense", "config": {"units": VOCAB, "useBias": True, "activation": "none"}, "inputShape": [HIDDEN], "outputShape": [VOCAB]},
        {"id": "softmax_3m", "name": "next character probabilities", "type": "softmax", "config": {}, "inputShape": [VOCAB], "outputShape": [VOCAB]},
    ]
    bundle = {
        "format": "neuronforge-standalone",
        "formatVersion": 1,
        "runtimeVersion": "1.3.0",
        "generator": "V0idGPT reproducible PyTorch trainer",
        "exportedAt": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "name": "v0idgpt-3-3M-reasoning",
        "task": "language",
        "architecture": {"inputShape": [SEQ], "inputName": "input", "layers": layers},
        "weights": weights,
        "output": {"shape": [VOCAB], "finalActivation": "softmax", "outputsAreProbabilities": True},
        "preprocessing": None,
        "image": None,
        "tokenizer": {"level": "char", "tokens": TOKENS, "merges": [], "seqLen": SEQ, "padToken": "<pad>", "oovRate": 0, "eos": "\n"},
        "sampling": {"temperature": 0.65, "topK": 8},
        "provenance": {
            "project": "V0idGPT",
            "dataset": "dataset.jsonl: balanced, authored conversational, factual, reasoning, and arithmetic examples",
            "loss": "assistant-only next-character cross_entropy",
            "trainedThisSession": True,
            "paramCount": param_count,
            "metrics": metrics,
            "seed": 3303,
        },
        "includes": {"inference": True, "training": False, "optimizerState": False, "dataset": True},
        "limitations": [
            "A 3.04M-parameter character model has limited language and factual coverage and can make mistakes.",
            "The model uses a bounded 128-character prompt context; it is not a general-purpose large language model.",
            "Outputs are learned statistical continuations, not guaranteed proofs or professional advice.",
        ],
        "tests": {
            "tolerance": {"atol": 0.0001, "rtol": 0.0001},
            "forward": [{"input": reference_input, "output": reference_output}],
            "tokenizer": [{"text": reference_prompt, "ids": reference_ids}],
            "generation": [],
            "preprocessing": [],
        },
    }
    text = "// Trained V0idGPT model: architecture, float32 weights, tokenizer, and provenance.\nexport default " + json.dumps(bundle, separators=(",", ":")) + ";\n"
    (MODEL_DIR / "bundle.mjs").write_text(text, encoding="utf-8")
    metrics_out = metrics | {"parameters": param_count, "examples": metrics["exampleCount"], "contextCharacters": SEQ}
    (MODEL_DIR / "training-metrics.json").write_text(json.dumps(metrics_out, indent=2) + "\n", encoding="utf-8")
    print(f"exported {param_count:,} parameters ({len(text) / 1e6:.1f} MB)", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=3303)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, args.threads))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    device = torch.device("cpu")
    examples = load_examples()
    all_targets = sum(int(sum(row["mask"])) for row in examples)
    categories = Counter(row["category"] for row in examples)
    order = list(range(len(examples)))
    random.Random(args.seed).shuffle(order)
    val_count = max(1, round(len(order) * 0.05))
    val_ids = set(order[:val_count])
    train = [row for i, row in enumerate(examples) if i not in val_ids]
    val = [row for i, row in enumerate(examples) if i in val_ids]
    train_targets = sum(int(sum(row["mask"])) for row in train)
    val_targets = sum(int(sum(row["mask"])) for row in val)

    model = Model().to(device)
    # Padding is an exact no-op in both the training batches and runtime prompts.
    with torch.no_grad():
        model.lstm.bias_ih_l0.zero_()
        model.lstm.bias_hh_l0.zero_()
    model.lstm.bias_ih_l0.requires_grad_(False)
    model.lstm.bias_hh_l0.requires_grad_(False)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr, weight_decay=0.005)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2, min_lr=0.00002)

    history = []
    best_state = None
    best_validation = float("inf")
    best_epoch = 0
    stale_epochs = 0
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        random.Random(args.seed + epoch).shuffle(train)
        model.train()
        total_loss = total_chars = 0.0
        batches = 0
        for at in range(0, len(train), args.batch_size):
            batch = train[at : at + args.batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss_sum, count = loss_for(model, batch, device)
            loss = loss_sum / count.clamp_min(1.0)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss_sum.detach())
            total_chars += float(count.detach())
            batches += 1
            if batches % 50 == 0:
                print(f"epoch {epoch} batch {batches}/{(len(train) + args.batch_size - 1) // args.batch_size} loss {total_loss / total_chars:.4f}", flush=True)

        model.eval()
        val_loss = val_chars = 0.0
        with torch.no_grad():
            for at in range(0, len(val), args.batch_size):
                loss_sum, count = loss_for(model, val[at : at + args.batch_size], device)
                val_loss += float(loss_sum)
                val_chars += float(count)
        validation_mean = val_loss / max(val_chars, 1)
        row = {"epoch": epoch, "trainLoss": round(total_loss / max(total_chars, 1), 6), "validationLoss": round(validation_mean, 6)}
        history.append(row)
        scheduler.step(validation_mean)
        if validation_mean < best_validation - 1e-6:
            best_validation = validation_mean
            best_epoch = epoch
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
        print(row | {"learningRate": optimizer.param_groups[0]["lr"]}, flush=True)
        if epoch >= 12 and stale_epochs >= 5:
            print(f"early stopping after {epoch} epochs; best validation epoch was {best_epoch}", flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    metrics = {
        "epochs": len(history),
        "requestedEpochs": args.epochs,
        "bestEpoch": best_epoch,
        "bestValidationLoss": round(best_validation, 6),
        "batchSize": args.batch_size,
        "sequenceLength": SEQ,
        "trainingObjective": "assistant-only next-character loss",
        "exampleCount": len(examples),
        "trainingExamples": len(train),
        "validationExamples": len(val),
        "trainingTargetCharacters": train_targets,
        "validationTargetCharacters": val_targets,
        "targetCharacters": all_targets,
        "categories": dict(sorted(categories.items())),
        "history": history,
        "elapsedSeconds": round(time.time() - started, 2),
    }
    export(model, metrics)


if __name__ == "__main__":
    main()
