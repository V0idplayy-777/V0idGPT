# V0idGPT

A pixel-faithful, production-ready recreation of the ChatGPT web interface — powered **entirely in your browser**. There is no backend, no API key, and no other AI: every reply is generated token by token from weights that ship in this repository.

![stack](https://img.shields.io/badge/model-v0idgpt--4--135M-black) ![params](https://img.shields.io/badge/parameters-134%2C515%2C008-black) ![runtime](https://img.shields.io/badge/inference-100%25%20in--browser-black) ![deps](https://img.shields.io/badge/dependencies-zero-black)

## Run it

```bash
node server.mjs          # → http://localhost:8000
# or: npm start
```

Any static file server works too (`python3 -m http.server`, `npx serve`, …), as long as `.mjs` files are served with a JavaScript MIME type. ES modules require `http://` — opening `index.html` from `file://` will not work.

## The default model: v0idgpt-4-135M

| | |
|---|---|
| Parameters | **134,515,008** (measured, and asserted by the verifier) |
| Architecture | 30-layer Llama transformer: RMSNorm, grouped-query attention (9 query / 3 KV heads), SwiGLU, RoPE θ=100000, tied embeddings |
| Tokenizer | byte-level **BPE, 49,152 subword tokens** (`models/v0idgpt-4-135M/tokenizer.json`) |
| Context | 8,192 positions supported; the app uses 4,096 by default |
| Weights | `model.bin` — int4 (group 64, fp16 scales) + int8 embeddings, ≈85 MB, fetched once and cached |
| Runtime | WebGPU compute shaders, with a bit-exact CPU fallback (`javascript/llm/`) |
| Base | fine-tuned from SmolLM2-135M-Instruct (Apache-2.0) on this repo's corpus |

It is a real transformer, not a lookup table: the chat log is encoded with the BPE tokenizer into a
ChatML prompt, the weights are quantised in the browser and run through WebGPU with a KV cache, and
each token is sampled from the resulting distribution (temperature / top-K / repetition penalty, all
adjustable in Settings). On load the app cross-checks the GPU kernels against the CPU reference
implementation and silently falls back if they disagree, so you are never shown wrong text without
knowing.

```bash
node models/v0idgpt-4-135M/verify.mjs   # params, tokenizer ids, reference logits, greedy decode
```

See [`models/v0idgpt-4-135M/README.md`](models/v0idgpt-4-135M/README.md) for the training recipe, the
dataset, quantisation numbers, and an honest list of what a 135M model can and cannot do.

### Honest expectations

This model is fluent, keeps the thread of a conversation, answers small-arithmetic questions
correctly, and changes tone when asked. It is still 135 million parameters: world knowledge is
shallow, multi-step reasoning is unreliable, and it will confidently invent specifics. It has no
memory across page loads and no tools. Every reply comes from the weights — nothing is pre-written
and no request ever leaves the browser.

### Earlier models

These remain selectable in **Settings → Model** for comparison, and each one is a genuine
from-scratch model trained inside this repository:

| Model | Params | Tokenizer | Context | Folder |
|---|---|---|---|---|
| **v0idgpt-4-135M** (default) | **134,515,008** | BPE, 49,152 tokens | 8,192 tokens | `models/v0idgpt-4-135M/` |
| v0idgpt-3-3M-reasoning | 3,038,759 | character-level, 71 tokens | 128 chars | `models/v0idgpt-3-3M-reasoning/` |
| v0idgpt-2-40k | 41,824 | BPE subword, 512 tokens | 16 tokens | `models/v0idgpt-2-40k/` |
| V0idGPT | 6,434 | character-level, 34 tokens | 64 chars | `javascript/` |

The 3M and smaller models stream one character at a time from their own trained weights; they are
demo-scale and frequently misunderstand prompts or get arithmetic wrong, which is exactly why the
135M transformer is the default.

Each model remembers its own Temperature / Top-K / length, and **Settings → Model → Preset** has one-tap Coherent / Balanced / Wild settings tuned per model. Under any reply, the token icon shows the raw output token by token — tap a token to see the alternatives the model considered and what the sampler actually drew. Token views live in memory only, so they're available for replies generated in the current session.

## Features

Everything in the interface works — there are no dead buttons:

- **Chat** — streaming replies, stop generation, regenerate, edit-and-resend user messages, copy, read-aloud (speech synthesis), like/dislike with persistence
- **Sidebar** — full chat history grouped by date (Today / Yesterday / Previous 7 days…), live search (`Ctrl/⌘ K`), rename, share, delete with undo, collapse/expand (`Ctrl/⌘ Shift S`)
- **Composer** — auto-resizing input, `Enter` to send / `Shift+Enter` for newline, voice dictation (where supported), attach `.txt/.md` files (button, or drag & drop), "Surprise me" prompts, clear chat
- **Header** — model menu with live specs, share chat (Web Share API → clipboard fallback), light/dark theme toggle
- **Settings** — theme (System/Light/Dark), streaming & tidy-formatting toggles, temperature / Top-K / max-length sliders, model integrity verification, export/import chats as JSON, delete all chats
- **Privacy** — chats persist only in `localStorage`; inference never leaves the device
- Fully responsive (mobile drawer sidebar), keyboard accessible, reduced-motion aware

## Project layout

```
index.html          app shell
css/styles.css      ChatGPT-style UI (light + dark)
js/app.js           chat logic, streaming sampler, persistence
vendor/             Lucide icons (ISC licensed, vendored for offline use)
server.mjs          zero-dependency static server
javascript/         the original, untouched NeuronForge model export
models/              additional standalone model packages
models/v0idgpt-4-135M/  the 135M transformer: model.bin, tokenizer, verifier, model card
models/v0idgpt-3-3M-reasoning/  trained 3.04M model, curated corpus, metrics
javascript/llm/     BPE tokenizer + WebGPU / CPU transformer engines
dataset.json        the original model's training dataset (as exported)
bundle.json         the original model's standalone bundle (as exported)
scripts/             reproducible dataset generation and training/export scripts
```

## Icons

All icons are [Lucide](https://lucide.dev) (`vendor/lucide.min.js`, ISC license — see `vendor/LUCIDE-LICENSE.txt`), referenced via `data-lucide` attributes exactly like the official Lucide web package.

## License

Apache-2.0 (see `LICENSE`). The Lucide icon library is ISC licensed.
