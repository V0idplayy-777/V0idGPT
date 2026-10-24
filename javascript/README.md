# V0idGPT — NeuronForge standalone model (JavaScript / TypeScript)

Exported from NeuronForge on 2026-10-21T08:25:33.790Z. This package runs the trained model **without NeuronForge**: it contains a dependency-free inference runtime (v1.3.0) that implements every layer of the network, the trained weights, and the input/output handling used during training.

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
cd v0idgpt
node verify.mjs     # should print OK
node example.mjs
```

```js
import { loadModel } from "./model.mjs";

const model = await loadModel();

// Generate text (temperature + top-k sampling; pass a seed for reproducible output)
const text = model.generate("hello there", { maxTokens: 120, temperature: 0.7, topK: 50 });
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
- **Input shape:** `[64]` (batch dimension excluded)
- **Output shape:** `[34]`, final activation `softmax`
- **Tokenizer:** char-level, 34 tokens, context 64, pad token `" "`, end-of-sequence `"\n"`
  - Text is lowercased; characters outside the vocabulary are skipped.
  - Prompts longer than 64 tokens are truncated to the most recent 64; shorter ones are left-padded.

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
