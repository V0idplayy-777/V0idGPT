import { loadModel } from "./model.mjs";

const model = await loadModel();
const report = model.verify();
for (const check of report.checks) {
  console.log(`[${check.status.toUpperCase().padEnd(4)}] ${check.name}: ${check.detail}`);
}

const checks = [];
const check = (ok, name, detail = "") => {
  checks.push(Boolean(ok));
  console.log(`[${ok ? "PASS" : "FAIL"}] ${name}${detail ? `: ${detail}` : ""}`);
};

check(model.info.name === "v0idgpt-3-3M-reasoning", "model identity");
check(model.info.paramCount === 3038759, "parameter count is exactly 3,038,759");
check(model.info.inputShape[0] === 128, "128-character prompt context");
check(model.info.tokenizer?.vocabSize === 71, "71-token character vocabulary");

const prompt = "user: hello\nassistant:";
const ordinary = model.predict(prompt).raw;
check(ordinary.length === 71, "forward pass produces 71 probabilities");

const session = model.createGenerationSession(prompt);
let fast = session.predict().raw;
let maxDiff = Math.max(...fast.map((value, i) => Math.abs(value - ordinary[i])));
check(maxDiff < 1e-5, "incremental decoder matches full forward pass", `max probability error ${maxDiff.toExponential(2)}`);

const suffix = " sure";
for (const id of model.tokenize(suffix)) session.push(id);
const updated = model.predict(prompt + suffix).raw;
fast = session.predict().raw;
maxDiff = Math.max(...fast.map((value, i) => Math.abs(value - updated[i])));
check(maxDiff < 1e-5, "incremental state updates match multi-character forward pass", `max probability error ${maxDiff.toExponential(2)}`);

const greeting = model.generate(prompt, { maxTokens: 80, temperature: 0, topK: 1, seed: 1, stop: ["\n"] }).trim();
check(/^(hello|hi|hey)\b/i.test(greeting) && !/feedback loop/i.test(greeting), "greeting stays on-topic", greeting);
const feedback = model.generate("user: what does a feedback loop mean?\nassistant:", { maxTokens: 100, temperature: 0, topK: 1, seed: 1, stop: ["\n"] }).trim();
check(/feedback loop/i.test(feedback) && /system/i.test(feedback), "feedback-loop question stays relevant", feedback);
const sources = model.generate("user: why should i check multiple sources?\nassistant:", { maxTokens: 100, temperature: 0, topK: 1, seed: 1, stop: ["\n"] }).trim();
check(/source|independent|claim/i.test(sources) && !/calendar|fossil|backup/i.test(sources), "source question stays relevant", sources);
const followupPrompt = "user: i have a job interview tomorrow\nassistant: That can feel like a lot. We can prepare one answer at a time.\nuser: what if i get nervous?\nassistant:";
const followup = model.generate(followupPrompt, { maxTokens: 100, temperature: 0, topK: 1, seed: 1, stop: ["\n"] }).trim();
check(/nervous|breath|interview|think/i.test(followup) && !/fossil|calendar|backup/i.test(followup), "follow-up uses the recent conversation", followup);
const repairPrompt = "user: hi\nassistant: A feedback loop occurs when a system's output influences its future behavior.\nuser: i was just greeting you\nassistant:";
const repair = model.generate(repairPrompt, { maxTokens: 80, temperature: 0, topK: 1, seed: 1, stop: ["\n"] }).trim();
check(/(you're right|i misunderstood|unrelated)/i.test(repair) && /(hi|hello)/i.test(repair) && !/feedback loop/i.test(repair), "misunderstanding is acknowledged and the greeting is answered", repair);
const arithmetic = model.generate("user: what is 6 times 8\nassistant:", { maxTokens: 60, temperature: 0, topK: 1, seed: 1, stop: ["\n"] }).trim();
check(/48/.test(arithmetic), "6 times 8 returns 48", arithmetic);

const sampledGreeting = model.generate(prompt, { maxTokens: 80, temperature: 0.45, topK: 8, seed: 3303, stop: ["\n"] }).trim();
check(/^(hello|hi|hey)\b/i.test(sampledGreeting) && !/feedback loop/i.test(sampledGreeting), "sampled greeting stays on-topic", sampledGreeting);
const sampledArithmetic = model.generate("user: what is 6 times 8\nassistant:", { maxTokens: 60, temperature: 0.45, topK: 8, seed: 3303, stop: ["\n"] }).trim();
check(/48/.test(sampledArithmetic), "sampled arithmetic returns 48", sampledArithmetic);
const sampledFeedback = model.generate("user: what does a feedback loop mean?\nassistant:", { maxTokens: 100, temperature: 0.45, topK: 8, seed: 3303, stop: ["\n"] }).trim();
check(/feedback loop/i.test(sampledFeedback) && /system/i.test(sampledFeedback), "balanced sampling stays on-topic", sampledFeedback);
const sampledSources = model.generate("user: why should i check multiple sources?\nassistant:", { maxTokens: 100, temperature: 0.45, topK: 8, seed: 3303, stop: ["\n"] }).trim();
check(/source|independent|claim/i.test(sampledSources), "sampled source answer is relevant", sampledSources);
const sampledFollowup = model.generate(followupPrompt, { maxTokens: 100, temperature: 0.45, topK: 8, seed: 3303, stop: ["\n"] }).trim();
check(/nervous|breath|interview|think/i.test(sampledFollowup), "sampled follow-up stays relevant", sampledFollowup);

const ok = report.ok && checks.every(Boolean);
console.log(ok ? "OK" : "FAILED");
if (!ok) process.exitCode = 1;
