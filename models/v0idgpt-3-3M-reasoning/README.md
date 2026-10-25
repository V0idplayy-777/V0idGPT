# v0idgpt-3-3M-reasoning

A fully local, trained **3,038,759-parameter** conversational language model for V0idGPT. It is a real neural next-character model: responses come from its learned float32 weights, not hard-coded answer routing or a remote service.

## Architecture

| Property | Value |
|---|---|
| Network | 128-wide character embedding → 800-unit LSTM → 71-way dense softmax |
| Parameters | 3,038,759 (3.04M) |
| Context | 48 characters |
| Dataset | 2,000 JSONL examples |
| Training | 6 epochs, AdamW, next-character cross-entropy |
| Final loss | 0.215852 train / 0.292305 validation |
| Runtime | NeuronForge standalone runtime 1.3.0 |

The dataset covers concise arithmetic traces, logic, factual explanations, planning, decisions, writing, safety, and supportive conversation. `dataset.jsonl` contains exactly one pure training record per line. `training-metrics.json` records the completed run.

## Use

```js
import { loadModel } from "./model.mjs";
const model = await loadModel();
const answer = model.generate("user: hello\nassistant:", {
  maxTokens: 100, temperature: 0.45, topK: 8, stop: ["\n"]
});
```

```bash
node verify.mjs
node example.mjs
```

## Reproduce

From the repository root, install PyTorch in an isolated environment, then run:

```bash
python3 scripts/generate_3m_dataset.py
python3 scripts/train_3m_model.py --epochs 6 --batch-size 24
```

The checked-in bundle already contains the completed trained weights; Python and PyTorch are not needed for inference.

## Honest limits

Three million parameters is extremely small for open-ended language modelling. This model can learn response structure and compact reasoning patterns, but it will still make arithmetic, factual, and consistency errors. Its 48-character window also limits conversational memory. Do not rely on it for medical, legal, financial, safety-critical, or other high-stakes decisions.
