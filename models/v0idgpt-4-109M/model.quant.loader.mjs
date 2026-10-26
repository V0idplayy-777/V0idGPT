// Quantized loader for v0idgpt-4-109M — fetches bundle.quant.mjs + weights-*.bin, dequantizes to float32
import { createModel, NF_RUNTIME_VERSION } from "./runtime.mjs";

let cached = null;
let cachedBundle = null;

function b64encode(bytes) {
  if (typeof Buffer !== "undefined") return Buffer.from(bytes).toString("base64");
  let bin = "";
  for (let i = 0; i < bytes.length; i++) bin += String.fromCharCode(bytes[i]);
  return btoa(bin);
}

function dequantizeUint8(quantBytes, min, scale, shape) {
  const n = quantBytes.length;
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = quantBytes[i] * scale + min;
  return out;
}

async function loadQuantized() {
  // Load manifest
  const manifestMod = await import("./bundle.quant.mjs");
  const bundle = manifestMod.default;
  const indexResp = await fetch(new URL("./weights-index.json", import.meta.url));
  if (!indexResp.ok) throw new Error("weights-index.json not found");
  const index = await indexResp.json();
  // Fetch all chunks
  const chunks = [];
  for (const c of index.chunks) {
    const resp = await fetch(new URL(`./${c.file}`, import.meta.url));
    if (!resp.ok) throw new Error(`Failed to fetch ${c.file}: ${resp.status}`);
    const buf = new Uint8Array(await resp.arrayBuffer());
    chunks.push({ names: c.tensors, data: buf });
  }
  // Reconstruct quantized bytes per tensor (in order of manifest's weight keys)
  // Chunks store concatenated tensors in order of blob list (sorted keys)
  const sortedKeys = Object.keys(bundle.weights).sort();
  // Build map from tensor name -> bytes by slicing chunks sequentially
  let offset = 0;
  // We need to know sizes per tensor to slice; we have them in bundle.weights[*].size
  // But chunks are stored as concatenated in sorted order, so we can just iterate
  const allBytes = new Uint8Array(index.chunks.reduce((s, c) => s + c.size, 0));
  let pos = 0;
  for (const c of chunks) { allBytes.set(c.data, pos); pos += c.data.length; }
  // Now slice per tensor
  const floatWeights = {};
  let cursor = 0;
  for (const k of sortedKeys) {
    const meta = bundle.weights[k];
    const size = meta.size;
    const qBytes = allBytes.subarray(cursor, cursor + size);
    cursor += size;
    const deq = dequantizeUint8(qBytes, meta.min, meta.scale, meta.shape);
    const raw = new Uint8Array(deq.buffer);
    floatWeights[k] = {
      shape: meta.shape,
      dtype: "float32",
      data: b64encode(raw),
      checksum: meta.checksum
    };
  }
  bundle.weights = floatWeights;
  // Remove quantization marker for runtime
  delete bundle.quantization;
  // Ensure provenance reflects dequantized
  return bundle;
}

async function loadFloat32() {
  const m = await import("./bundle.mjs");
  return m.default;
}

export async function loadModel() {
  if (cached) return cached;
  let bundle;
  try {
    // Try quantized first (production)
    bundle = await loadQuantized();
  } catch (e) {
    // Fallback to float32 bundle
    try {
      bundle = await loadFloat32();
    } catch (e2) {
      throw new Error(`v0idgpt-4-109M: no bundle available (quantized failed: ${e.message}, float32 failed: ${e2.message}). Training may still be in progress.`);
    }
  }
  cachedBundle = bundle;
  cached = createModel(bundle);
  return cached;
}

export function loadModelSync() {
  if (cached) return cached;
  throw new Error("v0idgpt-4-109M is async-only (needs fetch for quantized weights). Use loadModel()");
}

export const bundle = null; // async only
export { NF_RUNTIME_VERSION };
export default loadModel;
