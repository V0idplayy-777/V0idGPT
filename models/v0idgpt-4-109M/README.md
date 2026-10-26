# v0idgpt-4-109M

A **109,109,248-parameter** highly conversational BPE language model for V0idGPT. Generates every reply live from learned float32 weights in the browser — no API, no pre-made responses.

| Property | Value |
|---|---|
| Architecture | BPE embedding (8192×512) → 3× LSTM (2048 hidden) → dense (8192) → softmax |
| Parameters | 109,109,248 |
| Tokenizer | BPE subword, 8192 tokens, 512-token context window |
| Dataset | 67,892 lines highly conversational corpus (conversation, arithmetic, knowledge, logic, planning, writing, safety) |
| Training | Assistant-only cross-entropy, AdamW, 512-token context, BPE |
| Runtime | NeuronForge standalone v1.3.0 (zero-dependency, in-browser) |

The model is trained on recent `user:` / `assistant:` turns with a 512-token sliding window (best we can squeeze while keeping 100M+ params and browser performance). It uses a sub-word BPE tokenizer (no pre-made responses) and generates each token from the model's probability distribution.

```js
import { loadModel } from "./model.mjs";
const model = await loadModel();
console.log(model.generate("user: hello\nassistant:", { maxTokens: 60, temperature: 0.7, topK: 40 }));
```

```bash
node verify.mjs
node example.mjs
```

The dataset and training script are in `dataset.jsonl` and `scripts/train_v4_final.py`. The model is general-purpose, not specialized to math — it handles greetings, context, arithmetic, and diverse conversation.

## Training

```bash
python3 scripts/train_v4_final.py --epochs 3 --batch-size 12
```

## Limitations

109M is still small vs. large LLMs. It can be wrong, but it is far more coherent than the 3M model, stays on topic, handles "nothing" and "no" correctly, and gets basic arithmetic right (6×8=48).

