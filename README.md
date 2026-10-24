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

The app conditions the model the way it was trained (`q: <your message>? a:` plus the recent conversation tail — the tokenizer keeps the last 64 characters), then samples each next character with the same temperature + top-k sampler as `model.generate()`, stopping at the model's end-of-sequence token (`\n`). You can watch this happen live: replies stream in character by character, and **Settings → Model → Verify** replays NeuronForge's reference test vectors to prove the weights in your browser are bit-identical to the export.

Verify the model from the command line any time:

```bash
npm run verify-model     # → should print OK
```

### Honest expectations

V0idGPT is a *tiny* character-level model trained on ~8K characters. It has learned the shape of greetings, Q&A and simple facts, but its replies are often fragmentary or nonsensical — that's genuinely what this model produces, and this app shows you its real output rather than faking answers with a bigger model. Use the Temperature / Top-K sliders in **Settings → Model** to explore its behaviour (lower Top-K is noticeably more coherent).

## Models

Switch models in **Settings → Model** or from the model menu in the header. The choice is remembered.

| Model | Params | Tokenizer | Context | Folder |
|---|---|---|---|---|
| V0idGPT | 6,434 | character-level, 34 tokens | 64 chars | `javascript/` |
| v0idgpt-2-40k | 41,824 | BPE subword, 512 tokens | 16 tokens | `models/v0idgpt-2-40k/` |

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
dataset.json        the model's training dataset (as exported)
bundle.json         the model's standalone bundle (as exported)
```

## Icons

All icons are [Lucide](https://lucide.dev) (`vendor/lucide.min.js`, ISC license — see `vendor/LUCIDE-LICENSE.txt`), referenced via `data-lucide` attributes exactly like the official Lucide web package.

## License

Apache-2.0 (see `LICENSE`). The Lucide icon library is ISC licensed.
