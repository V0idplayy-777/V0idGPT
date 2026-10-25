# v0idgpt-3-3M-reasoning

A fully local, **3,038,759-parameter** character-level conversational model for V0idGPT. It generates each reply from learned float32 weights in the browser. There is no runtime prompt-to-answer lookup, API request, or network inference.

## Architecture and training

| Property | Value |
|---|---|
| Network | 128-wide character embedding → 800-unit LSTM → 71-way dense softmax |
| Parameters | 3,038,759 |
| Tokenizer | ASCII character-level, 71 tokens |
| Prompt context | 128 characters |
| Dataset | Reproducible, balanced JSONL corpus in `dataset.jsonl` |
| Training objective | Next-character cross-entropy on assistant continuations only |
| Runtime | NeuronForge standalone runtime 1.3.0 |

The app formats recent turns as `user: ...` and `assistant: ...`, matching the training examples. The model predicts the next character from the prompt and its generated continuation. A small incremental LSTM decoder carries hidden state forward between characters; this avoids rerunning the full recurrent network for every character and makes streaming substantially faster.

The corpus contains varied greetings and multi-turn dialogue, concise factual explanations, small arithmetic and logic demonstrations, planning and writing help, and safety examples. Repeated filler examples have been removed. `training-metrics.json` records the latest run; `scripts/generate_3m_dataset.py` and `scripts/train_3m_model.py` reproduce the data and weights.

## Use

```js
import { loadModel } from "./model.mjs";
const model = await loadModel();
const session = model.createGenerationSession("user: hello\nassistant:");
const probabilities = session.predict().raw;
// After sampling a character ID, call session.push(characterId) before the next step.
```

The general NeuronForge generation API is also available:

```js
const answer = model.generate("user: hello\nassistant:", {
  maxTokens: 80, temperature: 0.2, topK: 4, stop: ["\n"]
});
```

```bash
node verify.mjs
node example.mjs
```

## Reproduce

Install PyTorch in an isolated Python environment, then run from the repository root:

```bash
python3 scripts/generate_3m_dataset.py
python3 scripts/train_3m_model.py --epochs 20 --batch-size 24
```

PyTorch is needed for training only. The checked-in bundle contains the completed weights; inference has no Python or external dependencies.

## Limitations

Three million parameters is very small for open-ended language modelling. The model can produce a friendly, relevant continuation for familiar patterns and handle some small problems, but it can still misunderstand, repeat itself, or give incorrect arithmetic and facts. It is not equivalent to a modern large language model and should not be relied on for medical, legal, financial, safety-critical, or other high-stakes decisions.
