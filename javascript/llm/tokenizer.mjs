/* Byte-level BPE tokenizer for v0idgpt-4-135M (SmolLM2 / GPT-2 style).
 *
 * Reproduces tokenizers' Sequence[Digits(individual_digits), ByteLevel(use_regex)] pipeline:
 *   1. split digit runs into single characters
 *   2. GPT-2 style regex split of each piece
 *   3. byte-level encoding
 *   4. greedy BPE merges by rank
 * Plus the file's added_tokens (<|im_start|> etc.), which are matched before BPE.
 *
 * Verified token-for-token against the Rust `tokenizers` library — see scripts/verify_tokenizer.mjs.
 */

const GPT2_PATTERN =
  /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;

function bytesToUnicode() {
  const bs = [];
  for (let i = 33; i <= 126; i++) bs.push(i);
  for (let i = 161; i <= 172; i++) bs.push(i);
  for (let i = 174; i <= 255; i++) bs.push(i);
  const cs = [...bs];
  let n = 0;
  for (let b = 0; b < 256; b++) {
    if (!bs.includes(b)) {
      bs.push(b);
      cs.push(256 + n);
      n++;
    }
  }
  const map = new Map();
  for (let i = 0; i < bs.length; i++) map.set(bs[i], String.fromCodePoint(cs[i]));
  return map;
}

const BYTE_ENCODER = bytesToUnicode();
const BYTE_DECODER = new Map([...BYTE_ENCODER].map(([b, c]) => [c, b]));

/** Split text into pre-token chunks the way Digits(individual) + ByteLevel(use_regex) does. */
function pretokenize(text) {
  // 1. split digit runs into individual digits, keeping everything else intact
  const parts = [];
  let cur = "";
  for (const ch of text) {
    if (ch >= "0" && ch <= "9") {
      if (cur) { parts.push(cur); cur = ""; }
      parts.push(ch);
    } else {
      cur += ch;
    }
  }
  if (cur) parts.push(cur);
  // 2. GPT-2 regex within each part
  const chunks = [];
  for (const p of parts) {
    const m = p.match(GPT2_PATTERN);
    if (m) chunks.push(...m);
  }
  return chunks;
}

function byteEncode(str) {
  const bytes = new TextEncoder().encode(str);
  let out = "";
  for (const b of bytes) out += BYTE_ENCODER.get(b);
  return out;
}

export class BPETokenizer {
  constructor(json) {
    this.vocab = new Map();       // token string -> id
    this.idToToken = [];
    this.merges = new Map();      // "a\u0000b" -> rank
    this.specialTokens = new Map(); // literal text -> id
    this.specialById = new Map();

    const model = json.model;
    for (const [tokStr, id] of Object.entries(model.vocab)) {
      this.vocab.set(tokStr, id);
      this.idToToken[id] = tokStr;
    }
    let rank = 0;
    for (const m of model.merges) {
      const parts = typeof m === "string" ? m.split(" ") : m;
      this.merges.set(parts[0] + "\u0000" + parts[1], rank++);
    }
    for (const a of json.added_tokens || []) {
      this.specialTokens.set(a.content, a.id);
      this.specialById.set(a.id, a.content);
      this.vocab.set(a.content, a.id);
      this.idToToken[a.id] = a.content;
    }
    this.eot = this.specialTokens.get("<|endoftext|>") ?? 0;
    this.imStart = this.specialTokens.get("<|im_start|>");
    this.imEnd = this.specialTokens.get("<|im_end|>");
    // longest-first list, so multi-token specials win
    this.specialList = [...this.specialTokens.keys()].sort((a, b) => b.length - a.length);
    this.cache = new Map();
  }

  static async fromURL(url) {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`tokenizer fetch failed: ${res.status}`);
    return new BPETokenizer(await res.json());
  }

  /** BPE over one already byte-encoded chunk. */
  _bpe(chunk) {
    const cached = this.cache.get(chunk);
    if (cached) return cached;
    let word = Array.from(chunk);
    if (word.length > 1) {
      while (true) {
        let bestRank = Infinity, bestIdx = -1;
        for (let i = 0; i < word.length - 1; i++) {
          const r = this.merges.get(word[i] + "\u0000" + word[i + 1]);
          if (r !== undefined && r < bestRank) { bestRank = r; bestIdx = i; }
        }
        if (bestIdx < 0) break;
        const merged = word[bestIdx] + word[bestIdx + 1];
        word = [...word.slice(0, bestIdx), merged, ...word.slice(bestIdx + 2)];
      }
    }
    const ids = word.map((w) => {
      const id = this.vocab.get(w);
      if (id === undefined) throw new Error(`unknown token "${w}"`);
      return id;
    });
    if (this.cache.size < 60000) this.cache.set(chunk, ids);
    return ids;
  }

  encode(text, { allowSpecial = true } = {}) {
    const ids = [];
    if (!allowSpecial || this.specialList.length === 0) {
      for (const chunk of pretokenize(text)) ids.push(...this._bpe(byteEncode(chunk)));
      return ids;
    }
    // split around special tokens first
    let rest = text;
    while (rest.length) {
      let hit = null, hitAt = -1;
      for (const s of this.specialList) {
        const at = rest.indexOf(s);
        if (at >= 0 && (hitAt < 0 || at < hitAt)) { hitAt = at; hit = s; }
      }
      const head = hit ? rest.slice(0, hitAt) : rest;
      if (head) for (const chunk of pretokenize(head)) ids.push(...this._bpe(byteEncode(chunk)));
      if (!hit) break;
      ids.push(this.specialTokens.get(hit));
      rest = rest.slice(hitAt + hit.length);
    }
    return ids;
  }

  decodeBytes(ids) {
    let bytes = [];
    for (const id of ids) {
      const tok = this.idToToken[id];
      if (tok === undefined) continue;
      if (this.specialById.has(id)) continue; // specials render as nothing
      for (const ch of tok) {
        const b = BYTE_DECODER.get(ch);
        if (b !== undefined) bytes.push(b);
      }
    }
    return new Uint8Array(bytes);
  }

  decode(ids) {
    try {
      return new TextDecoder("utf-8", { fatal: false }).decode(this.decodeBytes(ids));
    } catch {
      return "";
    }
  }

  get vocabSize() { return this.idToToken.length; }
}
