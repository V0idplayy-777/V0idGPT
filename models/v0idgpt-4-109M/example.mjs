import { loadModel } from "./model.mjs";
const model = await loadModel();
console.log(model.generate("user: hello\nassistant:", { maxTokens: 60, temperature: 0.7, topK: 40 }));
console.log(model.generate("user: what is 6 times 8\nassistant:", { maxTokens: 20, temperature: 0, topK: 1 }));
