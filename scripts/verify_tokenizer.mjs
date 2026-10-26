/* Verify javascript/llm/tokenizer.mjs against reference ids produced by the Rust tokenizer.
   Usage: node scripts/verify_tokenizer.mjs <tokenizer.json> <reference.json>            */
import { readFileSync } from "node:fs";
import { BPETokenizer } from "../javascript/llm/tokenizer.mjs";

const [, , tokPath, refPath] = process.argv;
const tok = new BPETokenizer(JSON.parse(readFileSync(tokPath, "utf8")));
const ref = JSON.parse(readFileSync(refPath, "utf8"));

let ok = 0, bad = [];
for (const { text, ids } of ref) {
  let mine;
  try { mine = tok.encode(text); } catch (e) { bad.push([text, String(e)]); continue; }
  const same = mine.length === ids.length && mine.every((v, i) => v === ids[i]);
  if (same) ok++;
  else bad.push([text, `ref ${JSON.stringify(ids.slice(0, 12))} got ${JSON.stringify(mine.slice(0, 12))}`]);
}
// round trip
let rtBad = 0;
for (const { text, ids } of ref) {
  const viaMyIds = tok.decode(tok.encode(text));
  const viaRef = tok.decode(ids);
  if (viaMyIds !== viaRef) rtBad++;
}
console.log(`encode: ${ok}/${ref.length} exact`);
console.log(`decode: ${ref.length - rtBad}/${ref.length} identical to reference decode`);
for (const b of bad.slice(0, 8)) console.log("  MISMATCH", JSON.stringify(b[0]).slice(0, 90), "->", b[1]);
process.exit(bad.length ? 1 : 0);
