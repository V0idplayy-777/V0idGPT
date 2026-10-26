/* Verify the exported v0idgpt-4-135M browser bundle from the command line.
 *
 *   node models/v0idgpt-4-135M/verify.mjs
 *   node models/v0idgpt-4-135M/verify.mjs --bin /tmp/model.bin --ref /tmp/reference.json
 *
 * Checks, in order:
 *   1. header/parameter count and quantisation summary
 *   2. the byte-level BPE tokenizer reproduces the trainer's token ids
 *   3. the int4/int8 kernels reproduce the trainer's reference logits (torch float32 + numpy dequant)
 *   4. greedy decoding produces the reference continuation
 * Runs the CPU reference engine, so it needs no GPU.
 */
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { parseModel, CPUEngine } from "../../javascript/llm/engine-cpu.mjs";
import { BPETokenizer } from "../../javascript/llm/tokenizer.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
const arg = (name, fallback) => {
  const i = process.argv.indexOf(`--${name}`);
  return i > 0 ? process.argv[i + 1] : fallback;
};
const BIN = arg("bin", join(HERE, "model.bin"));
const REF = arg("ref", join(HERE, "reference.json"));
const TOK = arg("tok", join(HERE, "tokenizer.json"));

const red = (s) => `\x1b[31m${s}\x1b[0m`;
const green = (s) => `\x1b[32m${s}\x1b[0m`;
const dim = (s) => `\x1b[2m${s}\x1b[0m`;

function fail(msg) {
  console.log(red(`✗ ${msg}`));
  process.exitCode = 1;
}
function ok(msg) {
  console.log(green(`✓ ${msg}`));
}

const raw = readFileSync(BIN);
const ab = raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength);
const { header, payloadOffset, bytes } = parseModel(ab);
console.log(`model      ${header.name} — ${header.params.toLocaleString()} parameters`);
console.log(`quant      ${header.quant}`);
console.log(`context    ${header.config.max_position_embeddings} positions`);
console.log(`weights    ${(raw.byteLength / 1e6).toFixed(1)} MB on disk`);
if (header.params < 100_000_000 || header.params > 200_000_000) {
  fail(`parameter count ${header.params.toLocaleString()} is outside the 100M–200M band`);
} else {
  ok("parameter count inside the 100M–200M band");
}

const ref = JSON.parse(readFileSync(REF, "utf8"));
const tok = new BPETokenizer(JSON.parse(readFileSync(TOK, "utf8")));
const eng = new CPUEngine(ab, header, payloadOffset, bytes);

// ---- tokenizer
let tokOk = 0;
for (const c of ref.cases) {
  const mine = tok.encode(c.prompt);
  if (mine.length === c.ids.length && mine.every((v, i) => v === c.ids[i])) tokOk++;
}
if (tokOk === ref.cases.length) ok(`tokenizer matches the trainer on ${tokOk}/${ref.cases.length} cases`);
else fail(`tokenizer mismatch (${tokOk}/${ref.cases.length})`);

// ---- logits + greedy
let logitsOk = 0, greedyOk = 0, worst = 0;
for (const c of ref.cases) {
  eng.reset(1024); // the CPU engine owns its KV state; start every prompt from scratch
  let logits = null;
  for (let i = 0; i < c.ids.length; i++) logits = eng.step(c.ids[i]);
  for (const [idx, val] of c.top10) worst = Math.max(worst, Math.abs(logits[idx] - val));
  let best = 0;
  for (let i = 1; i < logits.length; i++) if (logits[i] > logits[best]) best = i;
  if (best === c.top10[0][0]) logitsOk++;
}
if (worst < 0.05 && logitsOk === ref.cases.length) {
  ok(`kernels match reference logits (max error ${worst.toExponential(2)}) on ${logitsOk}/${ref.cases.length} prompts`);
} else {
  fail(`logit mismatch: max error ${worst.toExponential(2)}, argmax ${logitsOk}/${ref.cases.length}`);
}

// ---- greedy continuation
for (const c of ref.cases) {
  eng.reset(1024);
  let logits = null;
  for (const id of c.ids) logits = eng.step(id);
  const gen = [];
  for (let i = 0; i < c.greedy.length; i++) {
    let best = 0;
    for (let k = 1; k < logits.length; k++) if (logits[k] > logits[best]) best = k;
    gen.push(best);
    logits = eng.step(best);
  }
  const same = gen.join(",") === c.greedy.join(",");
  if (same) greedyOk++;
  else console.log(dim(`   prompt ${JSON.stringify(c.prompt.slice(-40))} -> ${JSON.stringify(tok.decode(gen))} (reference ${JSON.stringify(c.greedy_text)})`));
}
if (greedyOk === ref.cases.length) ok(`greedy continuations match on ${greedyOk}/${ref.cases.length} prompts`);
else fail(`greedy continuations differ (${greedyOk}/${ref.cases.length})`);

console.log(process.exitCode ? red("\nverification FAILED") : green("\nverification passed"));
