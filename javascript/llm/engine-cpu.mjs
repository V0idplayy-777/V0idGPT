/* v0idgpt-4-135M — CPU (plain JavaScript) inference engine.
 *
 * This is the reference implementation: it runs the same arithmetic the WebGPU engine does, on
 * typed arrays, with no dependencies. It is slow (a few tokens/second) but it is what the model
 * is checked against, and it is the fallback when WebGPU is unavailable or fails its self-check.
 *
 * Weight format: see scripts/export_model.py. Matrices are int4 (two's complement, packed two per
 * byte, group-64 fp16 scales) or int8 (per-row fp16 scales); norms are fp16.
 */

const F16 = new Float32Array(1 << 16);
const F16I = new Int32Array(F16.buffer);
const F32 = new Float32Array(1);
const F32I = new Int32Array(F32.buffer);
function halfToFloat(h) {
  const s = (h & 0x8000) >> 15, e = (h & 0x7c00) >> 10, f = h & 0x03ff;
  if (e === 0) return (s ? -1 : 1) * Math.pow(2, -14) * (f / 1024);
  if (e === 0x1f) return f ? NaN : (s ? -Infinity : Infinity);
  return (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / 1024);
}

export function parseModel(buffer) {
  const bytes = new Uint8Array(buffer);
  const magic = new TextDecoder().decode(bytes.subarray(0, 8));
  if (magic !== "V0IDGPT4") throw new Error(`bad model magic: ${magic}`);
  const headerLen = new DataView(buffer).getUint32(8, true);
  const header = JSON.parse(new TextDecoder().decode(bytes.subarray(12, 12 + headerLen)));
  return { header, payloadOffset: 12 + headerLen, bytes };
}

export class CPUEngine {
  constructor(buffer, header, payloadOffset, bytes) {
    this.header = header;
    // head_dim is derived, not stored: hidden_size / num_attention_heads
    this.cfg = {
      ...header.config,
      head_dim: header.config.hidden_size / header.config.num_attention_heads,
    };
    this.base = payloadOffset;
    this.bytes = bytes;
    this.tensors = new Map();
    for (const t of header.tensors) this.tensors.set(t.name, t);
    this.ropeCache = new Float32Array(this.cfg.max_position_embeddings * this.cfg.hidden_size);
    this._buildRope();
  }

  _f16Array(off, n) {
    const out = new Float32Array(n);
    const dv = new DataView(this.bytes.buffer, this.bytes.byteOffset + this.base + off, n * 2);
    for (let i = 0; i < n; i++) out[i] = halfToFloat(dv.getUint16(i * 2, true));
    return out;
  }

  /** returns {kind, data:Int8Array|Uint8Array, scales:Float32Array, rows, cols, group} */
  tensor(name) {
    const t = this.tensors.get(name);
    if (!t) throw new Error(`missing tensor ${name}`);
    if (t._loaded) return t;
    const start = this.base + t.offset;
    if (t.kind === "f16") {
      t.data = this._f16Array(t.offset, t.shape.reduce((a, b) => a * b, 1));
    } else if (t.kind === "q8") {
      t.data = new Int8Array(this.bytes.buffer.slice(this.bytes.byteOffset + start,
        this.bytes.byteOffset + start + t.bytes));
      t.scales = this._f16Array(t.scales, t.scale_shape.reduce((a, b) => a * b, 1));
    } else {
      t.data = new Uint8Array(this.bytes.buffer.slice(this.bytes.byteOffset + start,
        this.bytes.byteOffset + start + t.bytes));
      t.scales = this._f16Array(t.scales, t.scale_shape.reduce((a, b) => a * b, 1));
      t.group = t.group || 64;
    }
    t.rows = t.shape[0];
    t.cols = t.shape[1];
    t._loaded = true;
    return t;
  }

  _buildRope() {
    const { head_dim, rope_theta, max_position_embeddings } = this.cfg;
    this.cos = new Float32Array(max_position_embeddings * head_dim);
    this.sin = new Float32Array(max_position_embeddings * head_dim);
    for (let p = 0; p < max_position_embeddings; p++) {
      for (let i = 0; i < head_dim / 2; i++) {
        const inv = 1 / Math.pow(rope_theta, (2 * i) / head_dim);
        const a = p * inv;
        const c = Math.cos(a), s = Math.sin(a);
        this.cos[p * head_dim + i] = c;
        this.cos[p * head_dim + i + head_dim / 2] = c;
        this.sin[p * head_dim + i] = s;
        this.sin[p * head_dim + i + head_dim / 2] = s;
      }
    }
  }

  /** y = W @ x  for a quantised [rows, cols] matrix. */
  matvec(name, x, y) {
    const t = this.tensor(name);
    const { rows, cols } = t;
    if (!y) y = new Float32Array(rows);
    if (t.kind === "q4") {
      const g = t.group;
      const nG = (cols + g - 1) / g | 0;
      const halfCols = (cols + 1) >> 1;
      const data = t.data, sc = t.scales;
      for (let r = 0; r < rows; r++) {
        const rowBase = r * halfCols;
        const scBase = r * nG;
        let acc = 0;
        for (let gi = 0; gi < nG; gi++) {
          const kStart = gi * g;
          const kEnd = Math.min(kStart + g, cols);
          let partial = 0;
          let k = kStart;
          for (; k + 1 < kEnd; k += 2) {
            const byte = data[rowBase + (k >> 1)];
            const lo = (byte << 28) >> 28;        // low nibble  = even index
            const hi = (byte << 24) >> 28;        // high nibble = odd index
            partial += lo * x[k] + hi * x[k + 1];
          }
          if (k < kEnd) partial += ((data[rowBase + (k >> 1)] << 28) >> 28) * x[k];
          acc += partial * sc[scBase + gi];
        }
        y[r] = acc;
      }
    } else if (t.kind === "q8") {
      const data = t.data, sc = t.scales;
      for (let r = 0; r < rows; r++) {
        const base = r * cols;
        let acc = 0;
        for (let k = 0; k < cols; k++) acc += data[base + k] * x[k];
        y[r] = acc * sc[r];
      }
    } else {
      const data = t.data;
      for (let r = 0; r < rows; r++) {
        const base = r * cols;
        let acc = 0;
        for (let k = 0; k < cols; k++) acc += data[base + k] * x[k];
        y[r] = acc;
      }
    }
    return y;
  }

  /** embedding row lookup (input embedding); also used for the tied output projection. */
  embedRow(id) {
    const t = this.tensor("embed_tokens.weight");
    const C = t.cols;
    const out = new Float32Array(C);
    if (t.kind === "q4") {
      const g = t.group, nG = Math.ceil(C / g);
      const rowBase = id * Math.ceil(C / 2), scBase = id * nG;
      for (let gi = 0; gi < nG; gi++) {
        const s = t.scales[scBase + gi];
        for (let j = 0; j < g; j++) {
          const k = gi * g + j;
          if (k >= C) break;
          const byte = t.data[rowBase + (k >> 1)];
          const v = (k & 1) ? ((byte << 24) >> 28) : ((byte << 28) >> 28);
          out[k] = v * s;
        }
      }
    } else if (t.kind === "q8") {
      const s = t.scales[id];
      for (let k = 0; k < C; k++) out[k] = t.data[id * C + k] * s;
    } else {
      for (let k = 0; k < C; k++) out[k] = t.data[id * C + k];
    }
    return out;
  }

  /** logits = E @ h  (tied lm_head), computed without materialising E. */
  logitsFromHidden(h) {
    const t = this.tensor("embed_tokens.weight");
    const V = t.rows, C = t.cols;
    const out = new Float32Array(V);
    if (t.kind === "q8") {
      for (let r = 0; r < V; r++) {
        const base = r * C, s = t.scales[r];
        let acc = 0;
        for (let k = 0; k < C; k++) acc += t.data[base + k] * h[k];
        out[r] = acc * s;
      }
    } else {
      const g = t.group, nG = Math.ceil(C / g);
      for (let r = 0; r < V; r++) {
        const rowBase = r * Math.ceil(C / 2), scBase = r * nG;
        let acc = 0;
        for (let gi = 0; gi < nG; gi++) {
          let p = 0;
          for (let j = 0; j < g; j++) {
            const k = gi * g + j;
            if (k >= C) break;
            const byte = t.data[rowBase + (k >> 1)];
            p += ((k & 1) ? ((byte << 24) >> 28) : ((byte << 28) >> 28)) * h[k];
          }
          acc += p * t.scales[scBase + gi];
        }
        out[r] = acc;
      }
    }
    return out;
  }

  rmsnorm(x, weightName, out) {
    const n = x.length;
    let ss = 0;
    for (let i = 0; i < n; i++) ss += x[i] * x[i];
    const inv = 1 / Math.sqrt(ss / n + this.cfg.rms_norm_eps);
    const w = this.tensor(weightName).data;
    for (let i = 0; i < n; i++) out[i] = x[i] * inv * w[i];
    return out;
  }

  /** Allocate KV cache for `maxLen` tokens. */
  createState(maxLen) {
    const { num_hidden_layers, num_key_value_heads, head_dim } = this.cfg;
    const kvDim = num_key_value_heads * head_dim;
    return {
      len: 0,
      maxLen,
      k: new Float32Array(num_hidden_layers * maxLen * kvDim),
      v: new Float32Array(num_hidden_layers * maxLen * kvDim),
      kvDim,
    };
  }

  /** Keep only the first n cached positions (prefix reuse between turns). */
  truncate(n) {
    if (this.state && n < this.state.len) this.state.len = n;
    return this;
  }

  /** Drop the KV cache (called before each new conversation turn). */
  reset(maxLen) {
    this.state = this.createState(maxLen || this.cfg.max_position_embeddings);
    return this;
  }

  /** Run one token; position is state.len. Returns logits (Float32Array[V]). */
  step(id, pos) {
    if (!this.state) this.reset();
    // the caller always feeds tokens sequentially; a rewind means a new sequence
    if (typeof pos === "number" && pos < this.state.len) this.state.len = pos;
    const state = this.state;
    const cfg = this.cfg;
    const C = cfg.hidden_size, H = cfg.num_attention_heads, KV = cfg.num_key_value_heads;
    const hd = cfg.head_dim, kvDim = KV * hd, inter = cfg.intermediate_size;
    const at = state.len;

    let x = this.embedRow(id);
    const h = new Float32Array(C);
    const q = new Float32Array(H * hd);
    const k = new Float32Array(kvDim);
    const v = new Float32Array(kvDim);
    const att = new Float32Array(H * hd);
    const gate = new Float32Array(inter);
    const up = new Float32Array(inter);
    const act = new Float32Array(inter);
    const tmp = new Float32Array(Math.max(C, inter));

    for (let L = 0; L < cfg.num_hidden_layers; L++) {
      const p = `layers.${L}.`;
      this.rmsnorm(x, p + "input_layernorm.weight", h);
      this.matvec(p + "self_attn.q_proj.weight", h, q);
      this.matvec(p + "self_attn.k_proj.weight", h, k);
      this.matvec(p + "self_attn.v_proj.weight", h, v);

      // rotary
      const cosBase = at * hd;
      for (let hh = 0; hh < H; hh++) {
        const off = hh * hd;
        for (let i = 0; i < hd / 2; i++) {
          const c = this.cos[cosBase + i], s = this.sin[cosBase + i];
          const a = q[off + i], b = q[off + i + hd / 2];
          q[off + i] = a * c - b * s;
          q[off + i + hd / 2] = b * c + a * s;
        }
      }
      for (let hh = 0; hh < KV; hh++) {
        const off = hh * hd;
        for (let i = 0; i < hd / 2; i++) {
          const c = this.cos[cosBase + i], s = this.sin[cosBase + i];
          const a = k[off + i], b = k[off + i + hd / 2];
          k[off + i] = a * c - b * s;
          k[off + i + hd / 2] = b * c + a * s;
        }
      }

      // append KV
      const kBase = (L * state.maxLen + at) * kvDim;
      state.k.set(k, kBase);
      state.v.set(v, kBase);

      // attention
      const groupSize = H / KV;
      const scale = 1 / Math.sqrt(hd);
      for (let hh = 0; hh < H; hh++) {
        const kvh = Math.floor(hh / groupSize);
        const qOff = hh * hd, kOff = kvh * hd;
        let maxv = -Infinity;
        const scores = new Float32Array(at + 1);
        for (let t = 0; t <= at; t++) {
          const base = (L * state.maxLen + t) * kvDim + kOff;
          let acc = 0;
          for (let i = 0; i < hd; i++) acc += q[qOff + i] * state.k[base + i];
          const s = acc * scale;
          scores[t] = s;
          if (s > maxv) maxv = s;
        }
        let sum = 0;
        for (let t = 0; t <= at; t++) { scores[t] = Math.exp(scores[t] - maxv); sum += scores[t]; }
        const invSum = 1 / sum;
        for (let i = 0; i < hd; i++) {
          let acc = 0;
          for (let t = 0; t <= at; t++) {
            const base = (L * state.maxLen + t) * kvDim + kOff;
            acc += scores[t] * state.v[base + i];
          }
          att[qOff + i] = acc * invSum;
        }
      }
      this.matvec(p + "self_attn.o_proj.weight", att, tmp);
      for (let i = 0; i < C; i++) x[i] += tmp[i];

      // MLP
      this.rmsnorm(x, p + "post_attention_layernorm.weight", h);
      this.matvec(p + "mlp.gate_proj.weight", h, gate);
      this.matvec(p + "mlp.up_proj.weight", h, up);
      for (let i = 0; i < inter; i++) {
        const g = gate[i];
        act[i] = (g / (1 + Math.exp(-g))) * up[i];
      }
      this.matvec(p + "mlp.down_proj.weight", act, tmp);
      for (let i = 0; i < C; i++) x[i] += tmp[i];
    }
    state.len = at + 1;
    this.rmsnorm(x, "norm.weight", h);
    return this.logitsFromHidden(h);
  }
}
