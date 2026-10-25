import { loadModel } from "./model.mjs";

const model = await loadModel();
const prompt = "user: why should sources be checked?\nassistant:";
console.log(model.generate(prompt, {
  maxTokens: 120,
  temperature: 0.45,
  topK: 8,
  seed: 3303,
  stop: ["\n"],
}));
console.log(model.info);
