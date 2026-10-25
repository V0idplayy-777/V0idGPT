import { loadModel } from "./model.mjs";

const model = await loadModel();
const report = model.verify();
const checks = [
  [model.info.name === "v0idgpt-3-3M-reasoning", "model identity"],
  [model.info.paramCount === 3038759, "parameter count is exactly 3,038,759"],
  [model.info.inputShape[0] === 48, "48-character context"],
  [model.info.tokenizer?.vocabSize === 71, "71-token character vocabulary"],
  [model.predict("user: hello\\nassistant:").raw.length === 71, "forward pass produces 71 probabilities"],
];
for (const c of report.checks) console.log(`[${c.status.toUpperCase().padEnd(4)}] ${c.name}: ${c.detail}`);
for (const [ok, name] of checks) console.log(`[${ok ? "PASS" : "FAIL"}] ${name}`);
const ok = report.ok && checks.every(([passed]) => passed);
console.log(ok ? "OK" : "FAILED");
if (!ok) process.exitCode = 1;
