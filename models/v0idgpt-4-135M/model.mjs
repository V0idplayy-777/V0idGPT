/* v0idgpt-4-135M — model facade used by the app.
 *
 *   const m = await loadModel({ onProgress });
 *   await m.chat(history, { temperature, topK, maxTokens, onDelta });
 *
 * Everything runs locally: the tokenizer is a byte-level BPE implemented in tokenizer.mjs, and the
 * transformer runs on WebGPU (engine-gpu.mjs) or, if that is unavailable or fails its self-check, on
 * the CPU reference engine (engine-cpu.mjs). No network calls except fetching this repo's own files.
 */
import { BPETokenizer } from "../../javascript/llm/tokenizer.mjs";
import { parseModel, CPUEngine } from "../../javascript/llm/engine-cpu.mjs";
import { GPUEngine } from "../../javascript/llm/engine-gpu.mjs";

export const META = {
  id: "v0idgpt-4-135M",
  name: "v0idgpt-4-135M",
  params: 134515008,
  base: "SmolLM2-135M-Instruct",
  system: "You are v0idgpt, a small assistant running locally in the browser. Answer what you're asked, " +
          "be concise, and admit uncertainty.",
};

const IM_START = "<|im_start|>";
const IM_END = "<|im_end|>";
const MODEL_URL = new URL("./model.bin", import.meta.url);
const TOKENIZER_URL = new URL("./tokenizer.json", import.meta.url);
const REFERENCE_URL = new URL("./reference.json", import.meta.url);

async function fetchWithProgress(url, onProgress, label) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${label}: HTTP ${res.status}`);
  const total = Number(res.headers.get("content-length")) || 0;
  if (!res.body || !total) return new Uint8Array(await res.arrayBuffer());
  const reader = res.body.getReader();
  const chunks = [];
  let got = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    got += value.length;
    onProgress?.(got / total);
  }
  const out = new Uint8Array(got);
  let at = 0;
  for (const c of chunks) { out.set(c, at); at += c.length; }
  return out;
}

/** Build the ChatML transcript the model was fine-tuned on. */
export function buildPrompt(messages, system = META.system) {
  let s = "";
  if (system) s += `${IM_START}system\n${system}${IM_END}\n`;
  for (const m of messages) {
    if (m.role !== "user" && m.role !== "assistant") continue;
    const content = String(m.content ?? "").replace(/\r/g, "").trim();
    if (!content) continue;
    s += `${IM_START}${m.role}\n${content}${IM_END}\n`;
  }
  return s + `${IM_START}assistant\n`;
}

/** Keep the newest turns that fit in the context budget; always keep the system prompt. */
export function trimHistory(messages, tok, budget, system = META.system) {
  const fits = (msgs) => tok.encode(buildPrompt(msgs, system), { allowSpecial: true }).length <= budget;
  const msgs = messages.filter((m) => (m.role === "user" || m.role === "assistant") && m.content);
  if (fits(msgs)) return msgs;
  let start = 0;
  while (start < msgs.length - 1 && !fits(msgs.slice(start))) start++;
  let out = msgs.slice(start);
  // a lone trailing user turn that is itself too long gets hard-truncated
  if (out.length && !fits(out)) {
    const ids = tok.encode(buildPrompt(out, system)).slice(-budget);
    const text = tok.decode(ids);
    out = [{ role: "user", content: text.replace(/^[^]*?user\n/, "") }];
  }
  return out;
}

/** top-k + temperature sampling with a light repetition penalty. */
export function makeSampler({ temperature = 0.6, topK = 40, repetitionPenalty = 1.05 } = {}) {
  return function sample(logits, recent) {
    const n = logits.length;
    if (!(temperature > 0.01)) {
      let b = 0;
      for (let i = 1; i < n; i++) if (logits[i] > logits[b]) b = i;
      return b;
    }
    const adj = new Float32Array(logits);
    if (repetitionPenalty > 1 && recent?.length) {
      const seen = new Set(recent.slice(-64));
      for (const id of seen) {
        if (adj[id] > 0) adj[id] /= repetitionPenalty;
        else adj[id] *= repetitionPenalty;
      }
    }
    let max = -Infinity;
    for (let i = 0; i < n; i++) if (adj[i] > max) max = adj[i];
    let best = 0;
    for (let i = 1; i < n; i++) if (adj[i] > adj[best]) best = i;
    const k = Math.max(1, Math.min(topK | 0 || n, n));
    // partial selection of the top-k logits
    const idx = Array.from({ length: n }, (_, i) => i);
    idx.sort((a, b) => adj[b] - adj[a]);
    const pool = idx.slice(0, k);
    let sum = 0;
    const probs = new Float32Array(pool.length);
    for (let j = 0; j < pool.length; j++) {
      const p = Math.exp((adj[pool[j]] - max) / temperature);
      probs[j] = p;
      sum += p;
    }
    let r = Math.random() * sum;
    for (let j = 0; j < pool.length; j++) {
      r -= probs[j];
      if (r <= 0) return pool[j];
    }
    return pool[pool.length - 1] ?? best;
  };
}

export async function loadModel({ onProgress = () => {}, maxContext, backend = "auto" } = {}) {
  onProgress({ stage: "tokenizer", fraction: 0 });
  const tokJson = await fetch(TOKENIZER_URL).then((r) => {
    if (!r.ok) throw new Error(`tokenizer: HTTP ${r.status}`);
    return r.json();
  });
  const tok = new BPETokenizer(tokJson);

  onProgress({ stage: "weights", fraction: 0 });
  const bin = await fetchWithProgress(MODEL_URL, (f) => onProgress({ stage: "weights", fraction: f }), "model.bin");
  const { header, payloadOffset, bytes } = parseModel(bin.buffer);
  if (header.params !== META.params) {
    console.warn(`[v0idgpt] header params ${header.params} != expected ${META.params}`);
  }
  onProgress({ stage: "build", fraction: 1 });

  const ctx = Math.min(header.config.max_position_embeddings, maxContext || 4096);
  const cpu = new CPUEngine(bin.buffer, header, payloadOffset, bytes);

  let sessionTokens = [];
  let engine = cpu;
  let usedBackend = "cpu";
  let gpuNote = null;
  if (backend !== "cpu") {
    try {
      if (await GPUEngine.available()) {
        const gpu = await GPUEngine.create(bin.buffer, header, payloadOffset, bytes, { maxContext: ctx });
        // Cross-check the GPU kernels against the CPU reference before using them for real replies.
        const check = await compareEngines(cpu, gpu, tok);
        if (check.ok) { engine = gpu; usedBackend = "webgpu"; }
        else gpuNote = `WebGPU output did not match the CPU reference (${check.detail}); using CPU.`;
        if (!check.ok) { try { gpu.dispose?.(); } catch { /* noop */ } }
      } else {
        gpuNote = "WebGPU is not available in this browser; using the CPU engine.";
      }
    } catch (err) {
      gpuNote = `WebGPU failed to start (${err.message}); using the CPU engine.`;
    }
  }

  const ctxSize = usedBackend === "webgpu" ? engine.maxContext : ctx;

  async function generate(messages, opts = {}) {
    const {
      temperature = 0.6, topK = 40, maxTokens = 220, onDelta, signal, system = META.system,
      onToken,
    } = opts;
    const budget = Math.max(64, Math.min(ctxSize, 8192) - Math.min(maxTokens, 512) - 8);
    const history = trimHistory(messages, tok, budget, system);
    const promptTokens = tok.encode(buildPrompt(history, system));
    const ids = promptTokens.slice(-budget);
    const sample = makeSampler({ temperature, topK, repetitionPenalty: 1.04 });

    // Reuse the KV cache for the part of the prompt we have already processed — on a multi-turn
    // chat that turns an 45-token re-prefill into a handful of new tokens.
    let keep = 0;
    const cached = sessionTokens;
    const limit = Math.min(ids.length, cached.length);
    while (keep < limit && ids[keep] === cached[keep]) keep++;
    engine.truncate?.(keep);
    sessionTokens = ids.slice(0, keep);
    let logits = null;
    for (let i = keep; i < ids.length; i++) {
      if (signal?.aborted) return { text: "", tokens: [], stopped: true };
      logits = await engine.step(ids[i], i);
      sessionTokens.push(ids[i]);
    }
    const produced = [];
    let text = "";
    const stopSet = new Set([tok.imEnd, tok.eot]);
    for (let step = 0; step < maxTokens; step++) {
      if (signal?.aborted) return { text, tokens: produced, stopped: true };
      const pos = ids.length + produced.length;
      if (pos >= ctxSize) break;
      const next = sample(logits, produced);
      if (stopSet.has(next)) break;
      produced.push(next);
      sessionTokens.push(next);
      onToken?.(next);
      const full = tok.decode(produced);
      const delta = full.slice(text.length);
      text = full;
      if (delta) onDelta?.(delta, text);
      logits = await engine.step(next, pos);
    }
    return { text, tokens: produced, stopped: false };
  }

  return {
    info: {
      ...META,
      // shapes the existing UI reads from every bundled model
      paramCount: META.params,
      tokenizer: { vocabSize: tok.vocabSize, kind: "byte-level BPE" },
      vocabSize: tok.vocabSize,
      contextSize: ctxSize,
      runtimeVersion: "4.0",
      exportedAt: "2026-10-25",
      backend: usedBackend,
      gpuNote,
      quant: header.quant,
      group: header.group,
      file: header.tensors.length,
    },
    tokenizer: tok,
    backend: usedBackend,
    get gpuNote() { return gpuNote; },
    chat: generate,
    /** Forget the cached conversation prefix. */
    resetSession() { sessionTokens = []; engine.reset?.(); },
    encode: (t) => tok.encode(t),
    decode: (i) => tok.decode(i),
    /** Verify weights + kernels; used by Settings → Model → Verify. */
    async verify(reference) {
      const checks = [];
      const ref = reference?.cases ? reference : await fetch(REFERENCE_URL).then((r) => r.json());
      const c = ref.cases[0];
      const cpuLogits = await runIds(cpu, c.ids);
      let worst = 0;
      for (const [idx, val] of c.top10) worst = Math.max(worst, Math.abs(cpuLogits[idx] - val));
      const argmax = argmaxOf(cpuLogits);
      checks.push({
        name: "tokenizer",
        status: tok.encode(c.prompt).join(",") === c.ids.join(",") ? "pass" : "fail",
        detail: `${c.ids.length} tokens`,
      });
      checks.push({
        name: "weights",
        status: worst < 0.05 && argmax === c.top10[0][0] ? "pass" : "fail",
        detail: `max logit error ${worst.toExponential(2)}, argmax ${argmax}`,
      });
      if (usedBackend === "webgpu") {
        const gpuLogits = await runIds(engine, c.ids);
        const gpuDiff = maxDiffInTop(cpuLogits, gpuLogits, c.top10);
        checks.push({
          name: "webgpu kernels",
          status: gpuDiff < 0.5 ? "pass" : "fail",
          detail: `max difference vs CPU ${gpuDiff.toExponential(2)}`,
        });
      } else {
        checks.push({ name: "backend", status: "pass", detail: "CPU reference engine" });
      }
      return { ok: checks.every((k) => k.status === "pass"), checks, runtimeVersion: "v0idgpt-gpu-1" };
    },
  };
}

async function runIds(engine, ids) {
  engine.reset?.();
  let logits = null;
  for (let i = 0; i < ids.length; i++) logits = await engine.step(ids[i], i);
  return logits;
}

function argmaxOf(a) {
  let b = 0;
  for (let i = 1; i < a.length; i++) if (a[i] > a[b]) b = i;
  return b;
}

function maxDiffInTop(a, b, top) {
  let worst = 0;
  for (const [idx] of top) worst = Math.max(worst, Math.abs(a[idx] - b[idx]));
  return worst;
}

async function compareEngines(cpu, gpu, tok) {
  try {
    const ids = tok.encode(buildPrompt([{ role: "user", content: "hi" }])).slice(-12);
    const a = await runIds(cpu, ids);
    const b = await runIds(gpu, ids);
    let worst = 0;
    for (let i = 0; i < a.length; i++) worst = Math.max(worst, Math.abs(a[i] - b[i]));
    const same = argmaxOf(a) === argmaxOf(b);
    return { ok: worst < 0.5 && same, detail: `max |Δlogit| ${worst.toExponential(2)}` };
  } catch (err) {
    return { ok: false, detail: err.message };
  }
}
