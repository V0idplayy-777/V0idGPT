import bundle from "./bundle.mjs";
import { createModel, NF_RUNTIME_VERSION } from "./runtime.mjs";
import { attachFastGeneration } from "./fast-generation.mjs";

let cached = null;

/** Synchronously build (once) and return the model. */
export function loadModelSync() {
  if (!cached) cached = attachFastGeneration(createModel(bundle));
  return cached;
}

/** Async for API symmetry with loaders that fetch weights; resolves immediately. */
export async function loadModel() {
  return loadModelSync();
}

export { bundle, createModel, NF_RUNTIME_VERSION };
export default loadModel;
