/* Cross-check the JavaScript engine against Python reference logits.
   Usage: node scripts/verify_engine.mjs <model.bin> <reference.json>                        */
import { readFileSync } from "node:fs";
import { parseModel, CPUEngine } from "../javascript/llm/engine-cpu.mjs";

const [, , binPath, refPath] = process.argv;
const buf = readFileSync(binPath);
const t0 = Date.now();
const { header, payloadOffset, bytes } = parseModel(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
console.log(`model: ${header.name} params=${header.params.toLocaleString()} quant="${header.quant}" load=${Date.now() - t0}ms`);
const eng = new CPUEngine(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength), header, payloadOffset, bytes);
const ref = JSON.parse(readFileSync(refPath, "utf8"));

let worst = 0, argmaxOk = 0;
for (const c of ref.cases) {
  const st = eng.createState(512);
  const ts = Date.now();
  let logits = null;
  for (const id of c.ids) logits = eng.step(id, st);
  const ms = Date.now() - ts;
  let best = 0;
  for (let i = 1; i < logits.length; i++) if (logits[i] > logits[best]) best = i;
  const refBest = c.top10[0][0];
  const refMax = c.last_logits_absmax;
  // compare the top-10 logits
  let maxDiff = 0;
  for (const [idx, val] of c.top10) maxDiff = Math.max(maxDiff, Math.abs(logits[idx] - val));
  worst = Math.max(worst, maxDiff);
  if (best === refBest) argmaxOk++;
  // greedy continuation
  const cont = [...c.ids];
  const gen = [];
  for (let i = 0; i < c.greedy.length; i++) {
    let b = 0;
    for (let k = 1; k < logits.length; k++) if (logits[k] > logits[b]) b = k;
    gen.push(b);
    cont.push(b);
    logits = eng.step(b, st);
  }
  const match = JSON.stringify(gen) === JSON.stringify(c.greedy);
  console.log(`  prompt ${c.ids.length} tok: maxlogit diff ${maxDiff.toExponential(2)} (|logits|~${refMax.toFixed(1)})  argmax ${best === refBest ? "ok" : "MISMATCH"}  greedy ${match ? "match" : `MISMATCH ${JSON.stringify(gen)} vs ${JSON.stringify(c.greedy)}`}  ${(ms / c.ids.length).toFixed(0)} ms/token`);
}
console.log(`\ntop-10 logit agreement: argmax ${argmaxOk}/${ref.cases.length}, worst abs diff ${worst.toExponential(2)}`);
process.exit(argmaxOk === ref.cases.length && worst < 0.05 ? 0 : 1);
