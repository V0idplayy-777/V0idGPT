# V0idGPT

A pixel-faithful, production-ready recreation of the ChatGPT web interface — powered **entirely in your browser** by the exact V0idGPT neural model bundled in this repository (`javascript/`). There is no backend, no API key, and no other AI: every reply is generated locally, character by character, by the included NeuronForge model.

![stack](https://img.shields.io/badge/model-NeuronForge%20V0idGPT-black) ![runtime](https://img.shields.io/badge/inference-100%25%20in--browser-black) ![deps](https://img.shields.io/badge/dependencies-zero-black)

## Run it

```bash
node server.mjs          # → http://localhost:8000
# or: npm start
```

Any static file server works too (`python3 -m http.server`, `npx serve`, …), as long as `.mjs` files are served with a JavaScript MIME type. ES modules require `http://` — opening `index.html` from `file://` will not work.

## The model (used exactly as exported)

| | |
|---|---|
| Architecture | embedding → 4-head causal attention → dense → softmax |
| Parameters | 6,434 |
| Tokenizer | character-level, 34 tokens, 64-character context |
| Runtime | NeuronForge standalone runtime v1.3.0 (`javascript/runtime.mjs`) |
| Training format | `q: …? a: …` conversations (see `dataset.json`) |

Each model uses its own training prompt format and tokenizer. The default 3M model is trained on recent `user:` / `assistant:` turns, uses a 128-character prompt window, and streams each next character from the model's probability distribution until its end-of-sequence token (`\n`). Its browser decoder carries the LSTM state forward between characters instead of recomputing the full prompt every time. No greeting or answer lookup runs at inference time. **Settings → Model → Verify** checks the exported model weights and runtime.

Verify the model from the command line any time:

```bash
npm run verify-model       # original 6K model
npm run verify-model-2     # 40K model
npm run verify-model-3     # trained 3.04M reasoning model
```

### Honest expectations

The default 3M model generates replies live from its trained neural weights; the app does not route prompts to pre-written answers. It now has a balanced, non-duplicated training set, answer-focused training, recent-turn context, and incremental decoding. It is still only 3M parameters and character-level, so it is not comparable to a modern large assistant and can still misunderstand prompts or get facts and arithmetic wrong. Use it for lightweight conversation and experiments, not high-stakes decisions. The older 6K model remains much more limited; its small corpus often produces fragmentary text.

## Models

Switch models in **Settings → Model** or from the model menu in the header. The choice is remembered.

| Model | Params | Tokenizer | Context | Folder |
|---|---|---|---|---|
| V0idGPT | 6,434 | character-level, 34 tokens | 64 chars | `javascript/` |
| v0idgpt-2-40k | 41,824 | BPE subword, 512 tokens | 16 tokens | `models/v0idgpt-2-40k/` |
| **v0idgpt-3-3M-reasoning** | **3,038,759** | character-level, 71 tokens | 128 chars | `models/v0idgpt-3-3M-reasoning/` |

`v0idgpt-3-3M-reasoning` is the default model for new users. It is an 800-unit LSTM trained with assistant-only next-character loss on a reproducible, balanced corpus of conversational, factual, reasoning, arithmetic, planning, writing, and safety examples. The trainer, corpus generator, and metrics are checked in. Its package README documents the model's limits.

Each model remembers its own Temperature / Top-K / length, and **Settings → Model → Preset** has one-tap Coherent / Balanced / Wild settings tuned per model. Under any reply, the token icon shows the raw output token by token — tap a token to see the alternatives the model considered and what the sampler actually drew. Token views live in memory only, so they're available for replies generated in the current session.

To add another export, drop its `javascript/` folder into `models/<name>/` and add an entry to `MODELS` at the top of `js/app.js`.

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
models/v0idgpt-3-3M-reasoning/  trained 3.04M model, curated corpus, metrics
dataset.json        the original model's training dataset (as exported)
bundle.json         the original model's standalone bundle (as exported)
scripts/             reproducible dataset generation and training/export scripts
```

## Icons

All icons are [Lucide](https://lucide.dev) (`vendor/lucide.min.js`, ISC license — see `vendor/LUCIDE-LICENSE.txt`), referenced via `data-lucide` attributes exactly like the official Lucide web package.

## License

Apache-2.0 (see `LICENSE`). The Lucide icon library is ISC licensed.
