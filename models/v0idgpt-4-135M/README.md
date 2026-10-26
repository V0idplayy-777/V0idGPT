# v0idgpt-4-135M

A 134,515,008-parameter decoder-only transformer that runs inside the browser tab. This folder is the
complete, self-contained model: tokenizer, quantised weights, reference logits, and the dataset and
scripts that produced it live one directory up in `scripts/`.

| | |
|---|---|
| Parameters | **134,515,008** |
| Architecture | 30 × Llama block (RMSNorm → GQA attention → RMSNorm → SwiGLU), RoPE θ=100000, tied embeddings |
| Shape | hidden 576, 9 query heads / 3 KV heads (head_dim 64), FFN 1536, vocab 49,152 |
| Tokenizer | byte-level BPE, 49,152 tokens (`tokenizer.json`), verified token-for-token against the trainer |
| Context | 8,192 positions (default 4,096 tokens in the app to keep KV memory modest) |
| Weights | `model.bin` — int4 symmetric, group 64, fp16 scales + int8 embeddings (≈85 MB) |
| Base model | [SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct) (Apache-2.0), fine-tuned here |
| Inference | WebGPU (`javascript/llm/engine-gpu.mjs`) with a CPU reference engine as fallback |

## What it is

This is a *fine-tune*, not a from-scratch pretrain. Pretraining 135M parameters needs hundreds of
billions of tokens; what this project does instead is start from a strong 135M instruct model and
teach it a specific job: be a fast, direct, locally-running conversational assistant with a
consistent voice, correct short-form arithmetic, and the ability to hold the thread of a conversation.

The fine-tune is full-parameter (not LoRA) on a hand-written + generated conversational corpus, with
loss on assistant tokens only.

## Training

Five stages, all full-parameter, all on CPU (2 cores, fp32, gradient checkpointing, ≈1,450 optimizer
steps, ≈3 hours). Loss is masked next-token cross-entropy on assistant tokens plus `<|im_end|>`;
system and user turns carry no loss.

| stage | data | steps | lr | outcome |
|---|---|---|---|---|
| 1. conversational SFT | `dataset.jsonl` — 8,128 convs / 472k tokens, 2 epochs | 996 | 2.5e-4 | fluent, on-topic chat; 6×8 but not 47+38 |
| 2. arithmetic curriculum | `gen_arithmetic_curriculum.py` — 3,708 systematic ops | 200 | 1.2e-4 | carry addition fixed, but small talk became numbers |
| 3. rebalance | 40% arithmetic / 60% conversational | 120 | 7e-5 | chat partially restored |
| 4. chat recovery | non-math conversations only | 43 | 3e-5 | number reflex gone |
| 5. fix-ups | `gen_fixups.py` — recompute-after-a-wrong-answer, borrow subtraction, chat | 90 | 2.5e-5 | shipped weights (`model.bin`) |

Stages 2–3 were cut short when the build box OOM-killed the trainer; each stage was resumed from the
last good checkpoint. Stage 5's full 104-step run finished but its final checkpoint could not be
written (disk full), so the shipped bundle is the step-90 weights. Everything above is in
`training-metrics.json`.

The two data generators are the interesting part:

* `scripts/gen_arithmetic_curriculum.py` covers the arithmetic *space* instead of sampling it — the
  full 2..12 times table, every single-digit operation, 1,400 two-digit additions (carry cases
  oversampled), 1,000 two-digit subtractions (borrow cases oversampled), and hundreds-scale sums with
  written decompositions.
* `scripts/gen_fixups.py` teaches recovery: a wrong answer, the user re-asking, then a corrected
  answer with an admission. The main corpus deliberately teaches the *opposite* reflex for correct
  answers (never agree with "2 + 2 = 5"), so this stage fills the gap where the model was simply
  repeating a mistake.

Both generators compute every answer in code and re-verify every emitted conversation before writing
it, so the corpus contains no wrong answers to copy.

## Measured quality

`training-metrics.json` carries the run log, the quantisation table and the quality gate. On the
shipped bundle (temperature 0.3, top-K 20):

```
48.                                        <- what is 6 times 8
85.                                        <- what's 47 + 38
125.                                       <- what's 68 + 57
44. 91 - 40 = 51, then 51 - 7 = 44.        <- 91 - 47
12 × 12 = 144.
56.                                        <- what's 8 times 7
Hey! Any plans for the weekend?            <- hi
I'm v0idgpt — a small model running locally in your browser.   <- what's your name
Let me redo it: 7 + 9 = 16. I had that wrong. My mistake.       <- "no, I said 7 + 9"
```

Quantisation cost, measured on 3,255 assistant tokens (fp32 → int4 group-64 with fp16 scales, int8
embeddings): perplexity 1.2031 → 1.2390 (+2.98%); int4 everywhere would be 1.2696 (+5.53%) and about
13 MB smaller. Absolute perplexity is optimistic because the corpus is small enough that every
conversation has been seen — the deltas are the meaningful part.

## Quantisation

`model.bin` stores every matrix as signed 4-bit values in groups of 64 along the input dimension with
one fp16 scale per group (4.25 bits/parameter), except the tied embedding/output matrix, which is int8
per row. Total 84.9 MB for 134.5M parameters. The alternative `--embed-bits 4` export is ≈72 MB at
+5.5% perplexity.

## Verifying it

```bash
node models/v0idgpt-4-135M/verify.mjs
```

Checks the parameter count is in range, that the JS tokenizer reproduces the trainer's ids, that the
int4/int8 kernels reproduce PyTorch float32 reference logits (max abs error ≈2e-5), and that greedy
decoding matches the reference continuation. In the app, **Settings → Model → Verify** runs the same
comparison in the browser, and the WebGPU backend is additionally cross-checked against the CPU
engine at load time — if they disagree the app silently falls back to CPU rather than show you wrong
text.

## Honest limits

* 135M parameters is small. It is fluent and it follows the conversation, but its world knowledge is
  shallow and it will invent specifics when pressed on facts it does not have.
* Reliable: greetings, small talk, staying on topic, two-digit addition and subtraction (including
  carries and borrows), the times tables, short explanations, tone control, and correcting itself
  when you re-ask a question it got wrong.
* Unreliable: three-digit mental arithmetic, word problems with more than one step, obscure facts,
  long multi-stage reasoning, anything requiring up-to-date information. It is also terse to a fault
  on open-ended follow-ups — "what should I practice first?" can come back as a single word.
* Fine-tuning on ~500k tokens cannot add much new knowledge; what it does add is behaviour — format,
  brevity, arithmetic discipline and the willingness to admit an error.
* It has no memory between page loads and no tools. It cannot browse, run code, or see files.
* The CPU fallback is roughly 3–8 tokens/second; WebGPU is much faster. The UI shows which backend
  is active under Settings → Model.
