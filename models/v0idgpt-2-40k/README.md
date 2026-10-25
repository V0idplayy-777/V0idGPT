# v0idgpt-2-40k — NeuronForge standalone model (JavaScript / TypeScript)

Exported from NeuronForge on 2026-10-24T21:01:09.193Z. This package runs the trained model **without NeuronForge**: it contains a dependency-free inference runtime (v1.3.0) that implements every layer of the network, the trained weights, and the input/output handling used during training.

## Files

| File | Purpose |
|---|---|
| `model.mjs` | Entry point: `loadModel()` |
| `runtime.mjs` | Inference engine: matmul, convolution, pooling, RNN, normalisation, activations, tokenizer, sampler |
| `bundle.mjs` | Architecture, float32 weights (base64, checksummed), preprocessing/tokenizer, test vectors |
| `model.d.mts` | TypeScript declarations |
| `verify.mjs` | Replays test vectors computed by NeuronForge |
| `example.mjs` | Runnable usage example |
| `tokenizer.json` | The exact training vocabulary (also embedded in bundle.mjs) |

## Quick start

```bash
cd v0idgpt-2-40k
node verify.mjs     # should print OK
node example.mjs
```

```js
import { loadModel } from "./model.mjs";

const model = await loadModel();

// Generate text (temperature + top-k sampling; pass a seed for reproducible output)
const text = model.generate("hello there", { maxTokens: 40, temperature: 0.7, topK: 50 });
console.log(text);

// Full details: generated ids/tokens, settings and why it stopped
const run = model.generateDetailed("hello there", { maxTokens: 20, seed: 42 });
console.log(run.fullText, run.stoppedBy);

// Next-token distribution
console.log(model.predict("hello there").top.slice(0, 5));
```

In a browser, serve the folder over HTTP (ES modules do not load from `file://`) and `import { loadModel } from "./model.mjs"`. Bundlers (Vite, webpack, esbuild) can import it directly.

## Inputs and outputs

- **Task:** `language`
- **Input shape:** `[16]` (batch dimension excluded)
- **Output shape:** `[512]`, final activation `softmax`
- **Tokenizer:** bpe-level, 512 tokens, context 16, pad token `" "`, no end-of-sequence token
  - Text is lowercased and split into whitespace / non-whitespace runs. The h,e,t,he,e,r,i,n,a,n,a,r,e,d,e,n,o,n,s,t,an,d,o,r,o,u,l,e,a,s,r,o,a,t,v,er,w,as,e,t,r,e,r,i,o,l,i,t,t,er,n,o,b,e,on,e,a,d,c,o,s,a,m,e,en,t,u,n,i,l,a,l,i,d,v,e,t,o,in,g,n,e,i,s,b,o,a,y,h,ar,t,e,o,w,l,i,q,u,d,ay,a,in,l,y,w,h,ver,y,o,k,t,h,f,or,r,a,e,.,k,et,c,h,e,s,p,l,l,d,u,r,a,g,f,ro,it,h,w,ith,u,t,b,y,w,in,m,b,ol,d,h,ad,or,n,i,st,qu,i,qui,et,k,e,ou,r,g,h,id,e,i,on,mb,er,for,e,s,u,s,he,p,ol,th,at,ou,s,ou,ld,no,t,i,r,gh,t,g,u,m,ar,il,l,wh,o,y,.,d,er,m,u,fro,m,b,r,ou,t,i,c,e,x,bo,ok,e,very,no,te,ar,d,ter,n,b,ut,he,r,note,book,one,.,a,st,l,an,m,orn,morn,ing,win,ter,co,m,d,ow,d,i,with,out,sa,me,e,c,pl,ain,ou,n,c,or,he,d,b,ri,p,at,d,y,m,o,s,.,ro,w,w,e,oun,t,s,h,c,ar,i,ent,v,en,t,ter,o,m,s,e,le,.,a,ir,w,or,s,s,a,c,d,ed,re,cor,s,ide,o,p,recor,ded,pl,an,wor,k,u,ri,ard,en,g,arden,c,uri,curi,ous,y,e,be,fore,br,ar,li,brar,d,ar,ar,row,an,c,t,il,st,u,un,til,v,i,a,b,a,ke,p,t,c,ro,g,re,is,hed,pol,ished,ri,ver,sh,op,ly,.,c,ri,ch,.,work,shop,al,l,le,ar,w,he,ain,.,m,ist,morning,.,d,g,co,p,wh,i,s,o,is,h,a,p,g,in,mar,ket,m,on,en,gin,ro,om,ill,ag,v,illag,u,st,fro,z,lear,n,froz,en,c,le,s,il,o,d,qu,e,b,our,har,bour,m,ount,un,der,e,ven,s,day,l,ow,g,et,s,m,o,od,c,har,d,e,sm,all,o,ver,ye,ar,the,m,w,ent,st,ion,who,le,f,in,the,y,dar,k,t,le,k,ed,p,u,lan,tern,n,u,ne,ver,v,o,en,c,s,har,villag,e,sa,id,l,t,bo,x,is,ed,nu,mber,t,un,w,al,n,arrow,ke,pt,learn,ed,w,ould,mo,ve,ar,gu,whi,le,a,f,af,ter,ic,e.,he,ld,d,id,ou,ght,argu,ed,it,e,to,ok,year,s,st,air,w,er,com,p,que,stion,di,sa,gre,ed,pol,ite,m,et,ne,l,p,ast,ra,ised,stair,s.,tun,nel,vo,ice.,wal,ked,whe,re,b,ought,bo,dy,dark,.,fore,st,disa,greed,fin,ish,polite,ly.,ad,ow,me,adow,char,d,or,chard,co,me,m,er,dow,n,k,en,mu,ch.,sa,y,say,ing,d,es,har,ve,harve,st,ne,ar,the,r,mist,ake,pu,z,puz,z,enc,e.,h,one,not,.,st,ood,co,st,s,ist,bri,dg,ly,,,co,ver,cover,.,le,tter,i,ed,j,our,jour,ne,no,body,y,ou,a,,,ol,in,ro,te,t,ab,vi,olin,w,rote,ac,on,be,acon,comp,as,s,ent,t,.,pat,tern,in,sist,insist,ed,win,dow,after,no,afterno,on,even,ing,so,me,be,side,cop,ied,m,mer,re,st,ag,ain.,b,ro,i,f,l,ad,des,cri,sent,ence.,be,come,c,ould,every,one,g,one.,afternoon,,,lad,der,some,where,wer,e,b,ed,descri,bed,or,y.,re,su,resu,lt,bro,ken,ide,a,,quiet,ly,,t,r,ex,plain,su,mmer,the,ory.,for,get,tab,le.,ut,ion,forget,table.,un,forgettable.,ec,ution,ex,ecution,g,ent,b,lan,blan,ket,com,plain,shar,ed,complain,t.,engin,e,g,en,mount,ain,s,p,p,ed,ra,ve,bri,ght,an,s,w,ar,c,old,gent,le,hone,st,pat,ient,war,m,gen,er,le,ss,sil,ent,b,orn,b,rave,gener,ous,b,born,stu,bborn,rest,less,day,.,l,at,c,al,d,dy,mu,ddy,anc,ient,o,f,librar,y,d,ust,dust,y,tr,ans,v,ed,cle,ver,ag,er,e,ager,cal,m,trans,lat,ok,ed,a,k,g,old,gold,en,u,l,cro,oked,g,a,li,ve,e,f,car,ef,caref,ul,puzz,le,live,ly,journe,y,i,ve,c,he,s,w,p,er,t,ur,s,un,ket,tle,if,t,s,er,sw,ift,bridg,e,c,e,compas,s,o,tter,ar,a,u,ar,ser,ved,b,en,t,u,a,m,w,ill,car,ri,e,p,t,ed,tu,es learned BPE merge rules are re-applied, so subword tokenisation matches training exactly; unseen characters become the `<unk>` token (id 0).
  - Prompts longer than 16 tokens are truncated to the most recent 16; shorter ones are left-padded.

## API

- `loadModel()` / `loadModelSync()` → model
- `model.predict(input, { preprocessed? })` → task-aware result (`predicted`, `label`, `probability`/`probabilities`, `value`, `reconstruction`, `top`, `raw` …)
- `model.predictBatch(inputs)`
- `model.forward(values, batchSize)` → `{ shape, data: Float32Array }` raw network output
- `model.tokenize(text)`, `model.detokenize(ids)`, `model.nextToken(text)`
- `model.generate(prompt, { maxTokens, temperature = 0.7, topK = 50, seed, stop, stopAtEos, includePrompt })`
- `model.generateDetailed(prompt, options)` → `{ text, fullText, ids, tokens, stoppedBy, settings }`
- `model.verify()` → `{ ok, checks }`
- `model.info`, `model.getWeights()`

## Numerical behaviour

The runtime mirrors NeuronForge's engine operation for operation: float32 storage, NHWC layout, the same loop order, GELU's tanh approximation, numerically stable sigmoid and softmax, Dropout disabled, BatchNorm using running statistics. In the same JavaScript engine results are normally bit-identical to NeuronForge; across engines `Math.exp`/`Math.tanh` can differ in the last digit, which is why `verify()` uses a tolerance (0.00001 abs + 0.0001 rel) and reports a sampling divergence as a warning rather than a failure.

On load the runtime validates every layer type, recomputes every tensor shape, checks each weight's shape and checksum (bound to its layer), and rejects unused tensors. Unsupported or corrupted bundles throw descriptive errors instead of producing wrong numbers.

## What this export does not include

- Newline ("\n") is treated as end-of-sequence by default (stopAtEos). Pass stop: [] to disable.
- Inference only: the export cannot train, fine-tune, or compute gradients, and contains no optimiser state. Use “Download checkpoint” to resume training inside NeuronForge.
- BatchNorm uses its running statistics and Dropout is disabled, matching NeuronForge's evaluation behaviour.
- Exporting preserves the learned behaviour; it does not make the model more capable.
- The training dataset is included as dataset.json (optional export).
- ONNX export is not implemented.
