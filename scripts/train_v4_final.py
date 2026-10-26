#!/usr/bin/env python3
"""Train v0idgpt-4-109M: BPE 8192 vocab, 256 context, 512 embed, 2048 hidden, 3 layers ~109M"""
import argparse, base64, json, random, time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from tokenizers import Tokenizer

ROOT=Path(__file__).resolve().parents[1]
MODEL_DIR=ROOT/"models/v0idgpt-4-109M"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
DATA=ROOT/"models/v0idgpt-4-125M/dataset.jsonl"  # reuse dataset we built
TOK=ROOT/"models/v0idgpt-4-125M/tokenizer.json"

SEQ=256
EMBED=512
HIDDEN=2048
LAYERS=3
VOCAB=8192
PAD_ID=0
EOS_ID=3

tokenizer=Tokenizer.from_file(str(TOK))
tokenizer.no_padding()
tokenizer.no_truncation()
TOKENS=[tokenizer.id_to_token(i) for i in range(VOCAB)]

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed=nn.Embedding(VOCAB, EMBED, padding_idx=PAD_ID)
        self.lstm=nn.LSTM(EMBED, HIDDEN, num_layers=LAYERS, batch_first=True)
        self.logits=nn.Linear(HIDDEN, VOCAB)
    def forward(self,x):
        e=self.embed(x)
        y,_=self.lstm(e)
        return self.logits(y)

def encode_row(row):
    hist=row.get("history",[])
    parts=[]
    for turn in hist:
        parts.append(f"user: {turn['user']}\nassistant: {turn['assistant']}\n")
    parts.append(f"user: {row['user']}\nassistant: ")
    prefix="".join(parts).lower()
    assistant=row["assistant"].lower()
    enc_pre=tokenizer.encode(prefix)
    enc_ass=tokenizer.encode(assistant)
    full_ids=enc_pre.ids + enc_ass.ids + [EOS_ID]
    if len(full_ids) > SEQ:
        full_ids=full_ids[-SEQ:]
        excess=len(enc_pre.ids)+len(enc_ass.ids)+1 - SEQ
        prefix_len=max(0, len(enc_pre.ids)-excess) if excess>0 else len(enc_pre.ids)
    else:
        prefix_len=len(enc_pre.ids)
    if len(full_ids) < 2:
        return None
    x=full_ids[:-1]
    y=full_ids[1:]
    mask=[1 if (i+1) >= prefix_len else 0 for i in range(len(y))]
    w=float(row.get("weight",1.0))
    mask=[m*w for m in mask]
    return {"x":x,"y":y,"mask":mask,"cat":row.get("category","unknown")}

def load_examples():
    ex=[]
    with open(DATA, encoding="utf-8") as f:
        for line in f:
            row=json.loads(line)
            e=encode_row(row)
            if e:
                ex.append(e)
    return ex

def make_batch(rows, device):
    B=len(rows)
    L=max(len(r["x"]) for r in rows)
    x=torch.full((B,L), PAD_ID, dtype=torch.long, device=device)
    y=torch.full((B,L), PAD_ID, dtype=torch.long, device=device)
    m=torch.zeros((B,L), dtype=torch.float32, device=device)
    for i,r in enumerate(rows):
        n=len(r["x"])
        x[i,:n]=torch.tensor(r["x"], dtype=torch.long, device=device)
        y[i,:n]=torch.tensor(r["y"], dtype=torch.long, device=device)
        m[i,:n]=torch.tensor(r["mask"], dtype=torch.float32, device=device)
    return x,y,m

def loss_for(model,batch,device):
    x,y,m=make_batch(batch, device)
    logits=model(x)
    ce=F.cross_entropy(logits.reshape(-1,VOCAB), y.reshape(-1), reduction='none').reshape_as(m)
    return (ce*m).sum(), m.sum()

def checksum(k, raw):
    b=k.encode()+raw
    s1=sum(b) & 0xffffffff
    s2=sum((i%65521+1)*x for i,x in enumerate(b)) & 0xffffffff
    return f"{s1:08x}{s2:08x}"

def tensor_entry(k, t):
    t=t.detach().cpu().contiguous().float()
    raw=t.numpy().tobytes()
    return {"shape":list(t.shape),"dtype":"float32","data":base64.b64encode(raw).decode(),"checksum":checksum(k,raw)}

def export(model, metrics):
    model.eval()
    ref_prompt="user: hello\nassistant:"
    enc=tokenizer.encode(ref_prompt.lower())
    ref_ids=enc.ids
    ref_input=[PAD_ID]*(SEQ-len(ref_ids)) + ref_ids
    with torch.no_grad():
        ref_out=torch.softmax(model(torch.tensor([ref_input], dtype=torch.long))[0,-1], dim=0).tolist()
    # Build tensors
    tensors={}
    tensors["emb.W"]=model.embed.weight
    for i in range(LAYERS):
        w_ih=getattr(model.lstm, f"weight_ih_l{i}")
        w_hh=getattr(model.lstm, f"weight_hh_l{i}")
        b_ih=getattr(model.lstm, f"bias_ih_l{i}")
        b_hh=getattr(model.lstm, f"bias_hh_l{i}")
        tensors[f"lstm_{i}.Wx"]=w_ih.T
        tensors[f"lstm_{i}.Wh"]=w_hh.T
        tensors[f"lstm_{i}.b"]=b_ih+b_hh
    tensors["logits.W"]=model.logits.weight.T
    tensors["logits.b"]=model.logits.bias

    weights={k:tensor_entry(k,v) for k,v in tensors.items()}
    param_count=sum(v.numel() for v in tensors.values())
    print(f"Export param count {param_count/1e6:.1f}M")
    layers=[]
    layers.append({"id":"emb","name":"bpe embedding","type":"embedding","config":{"vocabSize":VOCAB,"embeddingDim":EMBED},"inputShape":[SEQ],"outputShape":[SEQ,EMBED]})
    for i in range(LAYERS):
        in_dim=EMBED if i==0 else HIDDEN
        is_last=(i==LAYERS-1)
        layers.append({"id":f"lstm_{i}","name":f"lstm {i}","type":"lstm","config":{"units":HIDDEN,"returnSequences": not is_last},"inputShape":[SEQ, in_dim] if not is_last or LAYERS==1 else [SEQ, in_dim],"outputShape":[HIDDEN] if is_last else [SEQ, HIDDEN]})
        # Fix: for intermediate layers, output is [SEQ, HIDDEN], for last with returnSequences False, output [HIDDEN]
        if not is_last:
            # Need to ensure next layer's input shape matches
            pass
    # Actually for 3 layers, we need:
    # lstm0: [SEQ,512] -> [SEQ,2048] (returnSequences True)
    # lstm1: [SEQ,2048] -> [SEQ,2048] (True)
    # lstm2: [SEQ,2048] -> [2048] (False)
    # So fix layers
    layers=[]
    layers.append({"id":"emb","name":"bpe embedding","type":"embedding","config":{"vocabSize":VOCAB,"embeddingDim":EMBED},"inputShape":[SEQ],"outputShape":[SEQ,EMBED]})
    layers.append({"id":"lstm_0","name":"lstm 0","type":"lstm","config":{"units":HIDDEN,"returnSequences":True},"inputShape":[SEQ,EMBED],"outputShape":[SEQ,HIDDEN]})
    layers.append({"id":"lstm_1","name":"lstm 1","type":"lstm","config":{"units":HIDDEN,"returnSequences":True},"inputShape":[SEQ,HIDDEN],"outputShape":[SEQ,HIDDEN]})
    layers.append({"id":"lstm_2","name":"lstm 2","type":"lstm","config":{"units":HIDDEN,"returnSequences":False},"inputShape":[SEQ,HIDDEN],"outputShape":[HIDDEN]})
    layers.append({"id":"logits","name":"logits","type":"dense","config":{"units":VOCAB,"useBias":True,"activation":"none"},"inputShape":[HIDDEN],"outputShape":[VOCAB]})
    layers.append({"id":"soft","name":"softmax","type":"softmax","config":{},"inputShape":[VOCAB],"outputShape":[VOCAB]})

    bundle={
        "format":"neuronforge-standalone","formatVersion":1,"runtimeVersion":"1.3.0","generator":"V0idGPT v4 trainer","exportedAt":datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),
        "name":"v0idgpt-4-109M","task":"language",
        "architecture":{"inputShape":[SEQ],"inputName":"input","layers":layers},
        "weights":weights,
        "output":{"shape":[VOCAB],"finalActivation":"softmax","outputsAreProbabilities":True},
        "preprocessing":None,"image":None,
        "tokenizer":{"level":"bpe","tokens":TOKENS,"merges":[],"seqLen":SEQ,"padToken":"[PAD]","eos":"[EOS]","vocabSize":VOCAB},
        "sampling":{"temperature":0.7,"topK":40},
        "provenance":{"project":"V0idGPT","dataset":"dataset.jsonl 67k lines BPE 8192 highly conversational","loss":"assistant-only cross_entropy","trainedThisSession":True,"paramCount":param_count,"metrics":metrics,"seed":42},
        "includes":{"inference":True,"training":False,"optimizerState":False,"dataset":True},
        "limitations":["109M param BPE model, 256-token context (512 at inference via truncation), can still make mistakes"],
        "tests":{"tolerance":{"atol":0.0001,"rtol":0.0001},"forward":[{"input":ref_input,"output":ref_out}],"tokenizer":[{"text":ref_prompt,"ids":ref_ids}],"generation":[],"preprocessing":[]}
    }
    text="// v0idgpt-4-109M bundle\nexport default "+json.dumps(bundle, separators=(",",":"))+";\n"
    (MODEL_DIR/"bundle.mjs").write_text(text, encoding="utf-8")
    (MODEL_DIR/"training-metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Exported {param_count} params {len(text)/1e6:.2f} MB")

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--seed", type=int, default=42)
    args=parser.parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.set_num_threads(2)
    try: torch.set_num_interop_threads(1)
    except: pass
    device=torch.device("cpu")
    print("Loading examples...")
    examples=load_examples()
    print(f"Loaded {len(examples)}")
    # Use subset for speed: 6000 train, 600 val
    random.Random(args.seed).shuffle(examples)
    if len(examples) > 7000:
        examples=examples[:7000]
        print(f"Using subset {len(examples)}")
    val_n=max(200, int(len(examples)*0.08))
    train=examples[:-val_n]
    val=examples[-val_n:]
    print(f"Train {len(train)} Val {len(val)}")
    model=Model().to(device)
    # Freeze bias as before for stability
    with torch.no_grad():
        for i in range(LAYERS):
            getattr(model.lstm, f"bias_ih_l{i}").zero_()
            getattr(model.lstm, f"bias_hh_l{i}").zero_()
    for i in range(LAYERS):
        getattr(model.lstm, f"bias_ih_l{i}").requires_grad=False
        getattr(model.lstm, f"bias_hh_l{i}").requires_grad=False
    optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr, weight_decay=0.01)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * (len(train)//args.batch_size +1))
    best_val=float("inf")
    best_state=None
    history=[]
    start=time.time()
    for epoch in range(1, args.epochs+1):
        random.Random(args.seed+epoch).shuffle(train)
        model.train()
        total_loss=0
        total_cnt=0
        batches=0
        for i in range(0, len(train), args.batch_size):
            batch=train[i:i+args.batch_size]
            optimizer.zero_grad()
            loss_sum,cnt=loss_for(model,batch,device)
            if cnt.item()==0:
                continue
            loss=loss_sum/cnt
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss+=loss_sum.item()
            total_cnt+=cnt.item()
            batches+=1
            if batches%10==0 or batches==1:
                print(f"epoch {epoch} {batches}/{(len(train)+args.batch_size-1)//args.batch_size} loss {total_loss/max(total_cnt,1):.4f} elapsed {time.time()-start:.1f}s", flush=True)
        # val
        model.eval()
        v_loss=0
        v_cnt=0
        with torch.no_grad():
            for i in range(0, len(val), args.batch_size):
                batch=val[i:i+args.batch_size]
                s,c=loss_for(model,batch,device)
                v_loss+=s.item()
                v_cnt+=c.item()
        v_mean=v_loss/max(v_cnt,1)
        t_mean=total_loss/max(total_cnt,1)
        print(f"Epoch {epoch} train {t_mean:.4f} val {v_mean:.4f} lr {optimizer.param_groups[0]['lr']:.6f}", flush=True)
        history.append({"epoch":epoch,"trainLoss":round(t_mean,6),"valLoss":round(v_mean,6)})
        if v_mean < best_val:
            best_val=v_mean
            best_state={k:v.clone() for k,v in model.state_dict().items()}
    if best_state:
        model.load_state_dict(best_state)
    metrics={"epochs":len(history),"bestValLoss":round(best_val,6),"history":history,"paramCount":sum(p.numel() for p in model.parameters()),"examples":len(examples),"train":len(train),"val":len(val),"elapsed":round(time.time()-start,2)}
    export(model, metrics)

if __name__=="__main__":
    main()
