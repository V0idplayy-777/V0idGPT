#!/usr/bin/env python3
"""Train and export v0idgpt-3-3M-reasoning in NeuronForge's standalone format.

Requires PyTorch. The export is plain float32 and runs with the repository's
zero-dependency browser runtime; PyTorch is needed only to reproduce training.
"""
from __future__ import annotations
import argparse, base64, json, random, struct, time
from pathlib import Path
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models/v0idgpt-3-3M-reasoning"
DATA = MODEL_DIR / "dataset.jsonl"
SEQ, EMBED, HIDDEN = 48, 128, 800
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

def corpus() -> list[int]:
    ids: list[int] = []
    with DATA.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            text = f"user: {r['user']}\nassistant: {r['assistant']}\n".lower()
            # Dataset is intentionally ASCII-normalized during encoding.
            ids.extend(STOI.get(ch, STOI[" "]) for ch in text)
    return ids

def checksum(key: str, raw: bytes) -> str:
    all_bytes = key.encode() + raw
    s1 = sum(all_bytes) & 0xffffffff
    s2 = sum((i % 65521 + 1) * b for i, b in enumerate(all_bytes)) & 0xffffffff
    return f"{s1:08x}{s2:08x}"

def tensor_entry(key: str, t: torch.Tensor) -> dict:
    t = t.detach().cpu().contiguous().float()
    raw = t.numpy().tobytes()  # NumPy is bundled by current PyTorch wheels.
    return {"shape": list(t.shape), "dtype": "float32", "data": base64.b64encode(raw).decode(), "checksum": checksum(key, raw)}

def export(model: Model, metrics: dict) -> None:
    e, l, d = "embedding_3m", "lstm_3m", "logits_3m"
    # PyTorch gates and NeuronForge gates are both i,f,g,o. Runtime matrices are [in,out].
    tensors = {
        f"{e}.W": model.embed.weight,
        f"{l}.Wx": model.lstm.weight_ih_l0.T,
        f"{l}.Wh": model.lstm.weight_hh_l0.T,
        f"{l}.b": model.lstm.bias_ih_l0 + model.lstm.bias_hh_l0,
        f"{d}.W": model.logits.weight.T,
        f"{d}.b": model.logits.bias,
    }
    weights = {k: tensor_entry(k, v) for k, v in tensors.items()}
    param_count = sum(v.numel() for v in tensors.values())
    layers = [
      {"id":e,"name":"character embedding","type":"embedding","config":{"vocabSize":VOCAB,"embeddingDim":EMBED},"inputShape":[SEQ],"outputShape":[SEQ,EMBED]},
      {"id":l,"name":"reasoning memory","type":"lstm","config":{"units":HIDDEN,"returnSequences":False},"inputShape":[SEQ,EMBED],"outputShape":[HIDDEN]},
      {"id":d,"name":"next character logits","type":"dense","config":{"units":VOCAB,"useBias":True,"activation":"none"},"inputShape":[HIDDEN],"outputShape":[VOCAB]},
      {"id":"softmax_3m","name":"next character probabilities","type":"softmax","config":{},"inputShape":[VOCAB],"outputShape":[VOCAB]},
    ]
    bundle = {
      "format":"neuronforge-standalone","formatVersion":1,"runtimeVersion":"1.3.0",
      "generator":"V0idGPT reproducible PyTorch trainer","exportedAt":"2026-10-25T00:00:00.000Z",
      "name":"v0idgpt-3-3M-reasoning","task":"language",
      "architecture":{"inputShape":[SEQ],"inputName":"input","layers":layers},"weights":weights,
      "output":{"shape":[VOCAB],"finalActivation":"softmax","outputsAreProbabilities":True},
      "preprocessing":None,"image":None,
      "tokenizer":{"level":"char","tokens":TOKENS,"merges":[],"seqLen":SEQ,"padToken":"<pad>","oovRate":0,"eos":"\n"},
      "sampling":{"temperature":0.55,"topK":12},
      "provenance":{"project":"V0idGPT","dataset":"dataset.jsonl: 2,000 authored conversational and reasoning examples","loss":"cross_entropy","trainedThisSession":True,"paramCount":param_count,"metrics":metrics,"seed":3303},
      "includes":{"inference":True,"training":False,"optimizerState":False,"dataset":True},
      "limitations":["A 3.06M-parameter character model has limited factual coverage and may make mistakes.","The 48-character context window limits long conversations.","Outputs are learned statistical continuations, not guaranteed proofs or professional advice."],
      "tests":{"tolerance":0.0001,"forward":[],"tokenizer":[],"generation":[],"preprocessing":[]}
    }
    text = "// Trained V0idGPT model: architecture, float32 weights, tokenizer, and provenance.\nexport default " + json.dumps(bundle, separators=(",", ":")) + ";\n"
    (MODEL_DIR / "bundle.mjs").write_text(text, encoding="utf-8")
    (MODEL_DIR / "training-metrics.json").write_text(json.dumps(metrics | {"parameters":param_count,"examples":2000}, indent=2)+"\n")
    print(f"exported {param_count:,} parameters ({len(text)/1e6:.1f} MB)")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--lr", type=float, default=0.0015)
    args = ap.parse_args()
    random.seed(3303); torch.manual_seed(3303); torch.set_num_threads(max(1, min(16, torch.get_num_threads())))
    ids = corpus()
    # Independent windows reproduce the runtime's reset state and bounded context.
    windows = []
    for start in range(0, len(ids)-SEQ-1, SEQ):
        chunk = ids[start:start+SEQ+1]
        windows.append((chunk[:-1], chunk[1:]))
    split = int(len(windows)*0.95)
    train, val = windows[:split], windows[split:]
    model = Model()
    # Keeping recurrent biases at zero makes left-padding exactly state-neutral,
    # so browser inference matches training even for short prompts.
    with torch.no_grad():
        model.lstm.bias_ih_l0.zero_(); model.lstm.bias_hh_l0.zero_()
    model.lstm.bias_ih_l0.requires_grad_(False); model.lstm.bias_hh_l0.requires_grad_(False)
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss()
    history=[]; started=time.time()
    for epoch in range(1,args.epochs+1):
        random.Random(3303+epoch).shuffle(train); model.train(); total=n=0
        for at in range(0,len(train),args.batch_size):
            batch=train[at:at+args.batch_size]
            x=torch.tensor([z[0] for z in batch]); y=torch.tensor([z[1] for z in batch])
            opt.zero_grad(set_to_none=True); logits=model(x)
            loss=loss_fn(logits.reshape(-1,VOCAB),y.reshape(-1)); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
            total += loss.item(); n += 1
            if n%50==0: print(f"epoch {epoch} batch {n}/{(len(train)+args.batch_size-1)//args.batch_size} loss {total/n:.4f}",flush=True)
        model.eval(); vt=vn=0
        with torch.no_grad():
            for at in range(0,len(val),args.batch_size):
                b=val[at:at+args.batch_size]; x=torch.tensor([z[0] for z in b]); y=torch.tensor([z[1] for z in b])
                vt += loss_fn(model(x).reshape(-1,VOCAB),y.reshape(-1)).item(); vn += 1
        row={"epoch":epoch,"trainLoss":round(total/n,6),"validationLoss":round(vt/vn,6)}; history.append(row); print(row,flush=True)
    metrics={"epochs":args.epochs,"batchSize":args.batch_size,"sequenceLength":SEQ,"characterTokens":len(ids),"windows":len(windows),"history":history,"elapsedSeconds":round(time.time()-started,2)}
    export(model,metrics)

if __name__ == "__main__": main()
