import { loadModel } from "./model.mjs";

const model = await loadModel();
const report = model.verify();
for (const c of report.checks) console.log(`[${c.status.toUpperCase()}] ${c.name}: ${c.detail}`);
console.log(report.ok ? "OK" : "FAILED");
if (!report.ok) process.exitCode = 1;

// Additional checks
const prompt = "user: hello\nassistant:";
const out = model.generate(prompt, { maxTokens: 40, temperature: 0, topK: 1, seed: 1, stop: ["[EOS]"] });
console.log(`Greeting test: ${out.slice(0,80)}`);
const math = model.generate("user: what is 6 times 8\nassistant:", { maxTokens: 20, temperature: 0, topK: 1, seed: 1, stop: ["[EOS]"] });
console.log(`Math test 6*8: ${math}`);
if (!/48/.test(math)) {
  console.log("Math test FAILED: expected 48");
  process.exitCode = 1;
}
