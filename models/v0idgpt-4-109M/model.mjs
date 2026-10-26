import bundle from "./bundle.mjs";
import { createModel, NF_RUNTIME_VERSION } from "./runtime.mjs";

let cached = null;

export function loadModelSync() {
  if (!cached) cached = createModel(bundle);
  return cached;
}

export async function loadModel() {
  return loadModelSync();
}

export { bundle, createModel, NF_RUNTIME_VERSION };
export default loadModel;
