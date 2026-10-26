/* v0idgpt-4-135M — WebGPU inference engine.
 *
 * Decoder-only transformer, KV cache resident on the GPU. Weights live in three storage buffers
 * (packed 4-bit payload, fp32 group scales, fp32 norms); per token the only data that comes back to
 * the CPU is the 49,152 logits used for sampling.
 *
 * The maths mirrors javascript/llm/engine-cpu.mjs exactly (same dequantisation, rope, attention and
 * residual order), and the app cross-checks the two before trusting GPU output.
 *
 * Kernel notes
 *   mv_q4   16 rows per workgroup, 8 lanes per row; each lane owns whole 64-wide groups so the
 *           per-group scale is applied before the reduction.
 *   mv_q8   one lane per row, used for the tied output projection.
 *   attention  three passes (scores → softmax → weighted sum), 9 query heads over 3 KV heads.
 */

const SHADERS = /* wgsl */ `
struct Pm {
  a: u32, b: u32, c: u32, d: u32,
  e: u32, f: u32, g: u32, h: u32,
  x0: f32, x1: f32, x2: f32, x3: f32,
};
@group(0) @binding(0) var<uniform> pm: Pm;
@group(0) @binding(1) var<storage, read> W32: array<u32>;
@group(0) @binding(2) var<storage, read> SC: array<f32>;
@group(0) @binding(3) var<storage, read> IN: array<f32>;
@group(0) @binding(4) var<storage, read_write> OUT: array<f32>;
@group(0) @binding(5) var<storage, read_write> AUX: array<f32>;

var<workgroup> red: array<f32, 128>;

// ───────────────────────────────── rmsnorm ─────────────────────────────────
@compute @workgroup_size(128)
fn rmsnorm(@builtin(local_invocation_id) lid: vec3<u32>) {
  let t = lid.x;
  let n = pm.a;
  var ss: f32 = 0.0;
  var i: u32 = t;
  loop {
    if (i >= n) { break; }
    let v = IN[i];
    ss = ss + v * v;
    i = i + 128u;
  }
  red[t] = ss;
  workgroupBarrier();
  var stride: u32 = 64u;
  loop {
    if (stride == 0u) { break; }
    if (t < stride) { red[t] = red[t] + red[t + stride]; }
    workgroupBarrier();
    stride = stride >> 1u;
  }
  let inv = 1.0 / sqrt(red[0] / f32(n) + pm.x0);
  var j: u32 = t;
  loop {
    if (j >= n) { break; }
    OUT[j] = IN[j] * inv * SC[j];
    j = j + 128u;
  }
}

// ───────────────────────────────── int4 matvec ─────────────────────────────────
fn lane_partial(row: u32, lane: u32) -> f32 {
  let cols = pm.b;
  let group = pm.e;
  let nG = (cols + group - 1u) / group;
  let rowWords = (cols + 7u) / 8u;
  let wBase = pm.c + row * rowWords;
  let sBase = pm.d + row * nG;
  var acc: f32 = 0.0;
  var g: u32 = lane;
  loop {
    if (g >= nG) { break; }
    let s = SC[sBase + g];
    var part: f32 = 0.0;
    var k: u32 = g * group;
    let kend = min(k + group, cols);
    loop {
      if (k >= kend) { break; }
      let word = W32[wBase + (k >> 3u)];
      let bb = (word >> ((k & 7u) * 4u)) & 0xFu;
      let v: i32 = i32(bb << 28u) >> 28u;
      part = part + f32(v) * IN[k];
      k = k + 1u;
    }
    acc = acc + part * s;
    g = g + 8u;
  }
  return acc;
}

@compute @workgroup_size(128)
fn mv_q4(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let t = lid.x;
  let rowInWg = t / 8u;
  let lane = t % 8u;
  let row = wg.x * 16u + rowInWg;
  var acc: f32 = 0.0;
  if (row < pm.a) { acc = lane_partial(row, lane); }
  red[t] = acc;
  workgroupBarrier();
  if (lane < 4u) { red[t] = red[t] + red[t + 4u]; }
  workgroupBarrier();
  if (lane < 2u) { red[t] = red[t] + red[t + 2u]; }
  workgroupBarrier();
  if (lane < 1u) { red[t] = red[t] + red[t + 1u]; }
  workgroupBarrier();
  if (lane == 0u && row < pm.a) {
    if (pm.h == 1u) { OUT[row] = OUT[row] + red[t]; } else { OUT[row] = red[t]; }
  }
}

// ───────────────────────────────── int8 matvec ─────────────────────────────────
@compute @workgroup_size(128)
fn mv_q8(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let row = wg.x * 128u + lid.x;
  if (row >= pm.a) { return; }
  let cols = pm.b;
  let wordBase = pm.c / 4u + row * ((cols + 3u) / 4u);
  var acc: f32 = 0.0;
  var w: u32 = 0u;
  for (var k: u32 = 0u; k < cols; k = k + 1u) {
    if (k % 4u == 0u) { w = W32[wordBase + k / 4u]; }
    let byte = (w >> ((k % 4u) * 8u)) & 0xFFu;
    let v: i32 = i32(byte << 24u) >> 24u;
    acc = acc + f32(v) * IN[k];
  }
  let s = acc * SC[row];
  if (pm.h == 1u) { OUT[row] = OUT[row] + s; } else { OUT[row] = s; }
}

// ───────────────────────────────── embedding gather (int8) ─────────────────────────────────
@compute @workgroup_size(128)
fn embed_q8(@builtin(local_invocation_id) lid: vec3<u32>) {
  let cols = pm.a;
  let row = pm.b;
  let wordBase = pm.c / 4u + row * ((cols + 3u) / 4u);
  let s = SC[row];
  var k: u32 = lid.x;
  loop {
    if (k >= cols) { break; }
    let w = W32[wordBase + k / 4u];
    let byte = (w >> ((k % 4u) * 8u)) & 0xFFu;
    let v: i32 = i32(byte << 24u) >> 24u;
    OUT[k] = f32(v) * s;
    k = k + 128u;
  }
}

// ───────────────────────────────── rope (in place on q and k) ─────────────────────────────────
@compute @workgroup_size(64)
fn rope(@builtin(local_invocation_id) lid: vec3<u32>) {
  let i = lid.x;
  let heads = pm.a;
  let hd = pm.b;
  let half = hd / 2u;
  if (i >= half) { return; }
  let c = SC[pm.f + i];
  let s = SC[pm.g + i];
  for (var h: u32 = 0u; h < heads; h = h + 1u) {
    let base = h * hd;
    let a = OUT[base + i];
    let b = OUT[base + i + half];
    OUT[base + i] = a * c - b * s;
    OUT[base + i + half] = b * c + a * s;
    let a2 = AUX[base + i];
    let b2 = AUX[base + i + half];
    AUX[base + i] = a2 * c - b2 * s;
    AUX[base + i + half] = b2 * c + a2 * s;
  }
}

// ───────────────────────────────── copy vector into the KV cache ─────────────────────────────────
@compute @workgroup_size(128)
fn copy_vec(@builtin(local_invocation_id) lid: vec3<u32>) {
  var i: u32 = lid.x;
  loop {
    if (i >= pm.a) { break; }
    OUT[pm.b + i] = IN[i];
    i = i + 128u;
  }
}

// ───────────────────────────────── attention ─────────────────────────────────
@compute @workgroup_size(64)
fn attn_scores(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let head = wg.x;
  let p = wg.y * 64u + lid.x;
  if (p > pm.a) { return; }
  let hd = pm.b;
  let qBase = head * hd;
  let kBase = pm.d + p * pm.c;
  var acc: f32 = 0.0;
  for (var i: u32 = 0u; i < hd; i = i + 1u) {
    acc = acc + IN[qBase + i] * OUT[kBase + i];
  }
  AUX[head * 8192u + p] = acc * pm.x0;
}

@compute @workgroup_size(64)
fn attn_softmax(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let head = wg.x;
  let t = lid.x;
  let base = head * 8192u;
  let last = pm.a;
  var mx: f32 = -1e30;
  var i: u32 = t;
  loop {
    if (i > last) { break; }
    mx = max(mx, OUT[base + i]);
    i = i + 64u;
  }
  red[t] = mx;
  workgroupBarrier();
  var stride: u32 = 32u;
  loop {
    if (stride == 0u) { break; }
    if (t < stride) { red[t] = max(red[t], red[t + stride]); }
    workgroupBarrier();
    stride = stride >> 1u;
  }
  let maxv = red[0];
  var sum: f32 = 0.0;
  var j: u32 = t;
  loop {
    if (j > last) { break; }
    sum = sum + exp(OUT[base + j] - maxv);
    j = j + 64u;
  }
  red[t] = sum;
  workgroupBarrier();
  var s2: u32 = 32u;
  loop {
    if (s2 == 0u) { break; }
    if (t < s2) { red[t] = red[t] + red[t + s2]; }
    workgroupBarrier();
    s2 = s2 >> 1u;
  }
  let invSum = 1.0 / red[0];
  var k: u32 = t;
  loop {
    if (k > last) { break; }
    AUX[base + k] = exp(OUT[base + k] - maxv) * invSum;
    k = k + 64u;
  }
}

@compute @workgroup_size(64)
fn attn_weighted(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let head = wg.x;
  let i = lid.x;
  let hd = pm.b;
  if (i >= hd) { return; }
  var acc: f32 = 0.0;
  let last = pm.a;
  for (var p: u32 = 0u; p <= last; p = p + 1u) {
    acc = acc + AUX[head * 8192u + p] * IN[pm.d + p * pm.c + i];
  }
  OUT[head * hd + i] = acc;
}

// ───────────────────────────────── silu(gate) * up ─────────────────────────────────
@compute @workgroup_size(128)
fn silu_mul(@builtin(local_invocation_id) lid: vec3<u32>) {
  var i: u32 = lid.x;
  loop {
    if (i >= pm.a) { break; }
    let g = IN[i];
    OUT[i] = (g / (1.0 + exp(-g))) * AUX[i];
    i = i + 128u;
  }
}
`;

const SLOT = 256;          // uniform slot stride (256 = safe dynamic-offset alignment)

export class GPUEngine {
  static async available() {
    if (typeof navigator === "undefined" || !navigator.gpu) return false;
    try {
      const a = await navigator.gpu.requestAdapter();
      return !!a;
    } catch { return false; }
  }

  constructor(device, header, res) {
    this.device = device;
    this.header = header;
    this.res = res;
    const c = header.config;
    this.cfg = { ...c, head_dim: c.hidden_size / c.num_attention_heads };
  }

  static async create(buffer, header, payloadOffset, bytes, { maxContext = 4096 } = {}) {
    if (!navigator.gpu) throw new Error("WebGPU not available in this browser");
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error("no WebGPU adapter found");
    const device = await adapter.requestDevice();
    const ctx = Math.min(header.config.max_position_embeddings, maxContext);
    const res = await buildResources(device, header, payloadOffset, bytes, ctx);
    return new GPUEngine(device, header, res);
  }

  get maxContext() { return this.res.maxCtx; }

  /** Run one token at position `pos`; resolves to logits (Float32Array[vocab]). */
  async step(id, pos) {
    const d = this.device;
    const r = this.res;
    const cfg = this.cfg;
    const H = cfg.num_attention_heads, hd = cfg.head_dim, C = cfg.hidden_size;

    // 1. build every dispatch's uniform data up front (one writeBuffer for the whole token)
    const paramsBuf = new ArrayBuffer(r.slots * SLOT);
    const u32 = new Uint32Array(paramsBuf);
    const f32 = new Float32Array(paramsBuf);
    const calls = [];
    let slot = 0;
    const call = (pipe, bg, ints = [], floats = [], wx = 1, wy = 1) => {
      if (slot >= r.slots) throw new Error("dispatch ring overflow");
      const base = slot * (SLOT / 4);
      ints.forEach((v, i) => { u32[base + i] = v >>> 0; });
      floats.forEach((v, i) => { f32[base + 8 + i] = v; });
      calls.push({ pipe, bg, slot, wx, wy });
      slot++;
    };

    call(r.p.embed, r.bg.embed, [C, id, r.embedByteOff, 0], []);
    for (let L = 0; L < cfg.num_hidden_layers; L++) {
      const t = r.layers[L];
      call(r.p.rmsnorm, t.bg.rmsnorm1, [C], [cfg.rms_norm_eps]);
      call(r.p.mv4, t.bg.q, [C, C, t.q.w, t.q.s, 64], [], Math.ceil(C / 16));
      call(r.p.mv4, t.bg.k, [t.kvDim, C, t.k.w, t.k.s, 64], [], Math.ceil(t.kvDim / 16));
      call(r.p.mv4, t.bg.v, [t.kvDim, C, t.v.w, t.v.s, 64], [], Math.ceil(t.kvDim / 16));
      call(r.p.rope, t.bg.rope, [H, hd, 0, 0, 0, 0, pos * hd, r.sinOff + pos * hd], []);
      call(r.p.copy, t.bg.copyK, [t.kvDim, L * r.maxCtx * t.kvDim + pos * t.kvDim], []);
      call(r.p.copy, t.bg.copyV, [t.kvDim, L * r.maxCtx * t.kvDim + pos * t.kvDim], []);
      call(r.p.scores, t.bg.scores, [pos, hd, t.kvDim, L * r.maxCtx * t.kvDim], [1 / Math.sqrt(hd)],
        H, Math.ceil((pos + 1) / 64));
      call(r.p.softmax, t.bg.softmax, [pos], [], H);
      call(r.p.weighted, t.bg.weighted, [pos, hd, t.kvDim, L * r.maxCtx * t.kvDim], [], H);
      call(r.p.mv4, t.bg.o, [C, C, t.o.w, t.o.s, 64, 0, 0, 1], [], Math.ceil(C / 16));
      call(r.p.rmsnorm, t.bg.rmsnorm2, [C], [cfg.rms_norm_eps]);
      call(r.p.mv4, t.bg.gate, [t.inter, C, t.gate.w, t.gate.s, 64], [], Math.ceil(t.inter / 16));
      call(r.p.mv4, t.bg.up, [t.inter, C, t.up.w, t.up.s, 64], [], Math.ceil(t.inter / 16));
      call(r.p.silu, t.bg.silu, [t.inter], []);
      call(r.p.mv4, t.bg.down, [C, t.inter, t.down.w, t.down.s, 64, 0, 0, 1], [], Math.ceil(C / 16));
    }
    call(r.p.rmsnorm, r.bg.normOut, [C], [cfg.rms_norm_eps]);
    call(r.p.mv8, r.bg.logits, [cfg.vocab_size, C, r.embedByteOff, 0, 0, 0, 0, 0], [],
      Math.ceil(cfg.vocab_size / 128));

    d.queue.writeBuffer(r.uniform, 0, paramsBuf);

    // 2. encode
    const enc = d.createCommandEncoder();
    const pass = enc.beginComputePass();
    for (const c of calls) {
      pass.setPipeline(c.pipe);
      pass.setBindGroup(0, c.bg, [c.slot * SLOT]);
      pass.dispatchWorkgroups(c.wx, c.wy, 1);
    }
    pass.end();
    enc.copyBufferToBuffer(r.logitsBuf, 0, r.stage, 0, cfg.vocab_size * 4);
    d.queue.submit([enc.finish()]);

    await r.stage.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(r.stage.getMappedRange().slice(0));
    r.stage.unmap();
    return out;
  }
}

/* ───────────────────────────── resource construction ───────────────────────────── */
async function buildResources(device, header, payloadOffset, bytes, maxCtx) {
  const cfg = header.config;
  const C = cfg.hidden_size;
  const H = cfg.num_attention_heads;
  const kvDim = cfg.num_key_value_heads * (C / H);
  const inter = cfg.intermediate_size;
  const V = cfg.vocab_size;
  const L = cfg.num_hidden_layers;

  const tensors = new Map();
  for (const t of header.tensors) tensors.set(t.name, t);

  // ---- one arena holds every packed payload byte (u32 aligned by the exporter)
  let payloadBytes = 4;
  for (const t of header.tensors) payloadBytes = Math.max(payloadBytes, t.offset + t.bytes);
  const arena = new Uint32Array(Math.ceil(payloadBytes / 4));
  const arenaBytes = new Uint8Array(arena.buffer);
  const scaleVals = [], normVals = [];
  const scaleOff = new Map(), normOff = new Map();
  let sCur = 0, nCur = 0;

  for (const t of header.tensors) {
    arenaBytes.set(bytes.subarray(payloadOffset + t.offset, payloadOffset + t.offset + t.bytes), t.offset);
    const dvBase = bytes.byteOffset + payloadOffset;
    if (t.kind === "f16") {
      const n = t.shape.reduce((a, b) => a * b, 1);
      const dv = new DataView(bytes.buffer, dvBase + t.offset, t.bytes);
      normOff.set(t.name, nCur);
      for (let i = 0; i < n; i++) normVals.push(halfToFloat(dv.getUint16(i * 2, true)));
      nCur += n;
    } else {
      const n = t.scale_shape.reduce((a, b) => a * b, 1);
      const dv = new DataView(bytes.buffer, dvBase + t.scales, t.scale_bytes);
      scaleOff.set(t.name, sCur);
      for (let i = 0; i < n; i++) scaleVals.push(halfToFloat(dv.getUint16(i * 2, true)));
      sCur += n;
    }
  }

  const mkStore = (bytesLen) => device.createBuffer({
    size: Math.max(16, Math.ceil(bytesLen / 4) * 4),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const weightsBuf = mkStore(arena.byteLength);
  device.queue.writeBuffer(weightsBuf, 0, arena);
  const scalesBuf = mkStore(scaleVals.length * 4);
  device.queue.writeBuffer(scalesBuf, 0, new Float32Array(scaleVals));
  const normsBuf = mkStore(normVals.length * 4);
  device.queue.writeBuffer(normsBuf, 0, new Float32Array(normVals));

  const vec = (n) => mkStore(Math.max(64, n * 4));
  const xBuf = vec(C), hBuf = vec(C), qBuf = vec(C), kBuf = vec(kvDim), vBuf = vec(kvDim),
    attBuf = vec(C), gateBuf = vec(inter), upBuf = vec(inter), actBuf = vec(inter),
    logitsBuf = vec(V), scoresBuf = vec(8192 * H), probsBuf = vec(8192 * H);

  const kCache = vec(L * maxCtx * kvDim);
  const vCache = vec(L * maxCtx * kvDim);

  // rope tables
  const hd = C / H;
  const cos = new Float32Array(8192 * hd), sin = new Float32Array(8192 * hd);
  for (let p = 0; p < 8192; p++) {
    for (let i = 0; i < hd / 2; i++) {
      const a = p / Math.pow(cfg.rope_theta, (2 * i) / hd);
      const c = Math.cos(a), s = Math.sin(a);
      cos[p * hd + i] = c; cos[p * hd + i + hd / 2] = c;
      sin[p * hd + i] = s; sin[p * hd + i + hd / 2] = s;
    }
  }
  const ropeBuf = mkStore(cos.byteLength + sin.byteLength);
  device.queue.writeBuffer(ropeBuf, 0, cos);
  device.queue.writeBuffer(ropeBuf, cos.byteLength, sin);

  // ---- pipelines
  const mod = device.createShaderModule({ code: SHADERS });
  const bgl = device.createBindGroupLayout({
    entries: [0, 1, 2, 3, 4, 5].map((i) => ({
      binding: i,
      visibility: GPUShaderStage.COMPUTE,
      buffer: {
        type: i === 0 ? "uniform" : i >= 4 ? "storage" : "read-only-storage",
        hasDynamicOffset: i === 0,
        minBindingSize: 0,
      },
    })),
  });
  const layout = device.createPipelineLayout({ bindGroupLayouts: [bgl] });
  const pipe = (entryPoint) => device.createComputePipeline({ layout, compute: { module: mod, entryPoint } });
  const p = {
    rmsnorm: pipe("rmsnorm"), mv4: pipe("mv_q4"), mv8: pipe("mv_q8"), embed: pipe("embed_q8"),
    rope: pipe("rope"), copy: pipe("copy_vec"), scores: pipe("attn_scores"),
    softmax: pipe("attn_softmax"), weighted: pipe("attn_weighted"), silu: pipe("silu_mul"),
  };

  const slots = 720;
  const uniform = device.createBuffer({
    size: slots * SLOT, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const bg = (b1, b2, b3, b4, b5) => device.createBindGroup({
    layout: bgl,
    entries: [
      { binding: 0, resource: { buffer: uniform, offset: 0, size: 128 } },
      { binding: 1, resource: { buffer: b1 } },
      { binding: 2, resource: { buffer: b2 } },
      { binding: 3, resource: { buffer: b3 } },
      { binding: 4, resource: { buffer: b4 } },
      { binding: 5, resource: { buffer: b5 } },
    ],
  });

  const bgEmbed = bg(weightsBuf, scalesBuf, ropeBuf, xBuf, xBuf);
  const bgLogits = bg(weightsBuf, scalesBuf, hBuf, logitsBuf, logitsBuf);
  const bgNormOut = bg(weightsBuf, normsBuf, xBuf, hBuf, hBuf);

  // Shared bind groups. Layout: B1 = packed weights, B2 = scales (or a read-only dummy),
  // B3/B4/B5 = the buffers the kernel reads and writes; every kernel documents the slots it uses.
  const shared = {
    rmsnorm1: bg(weightsBuf, normsBuf, xBuf, hBuf, hBuf),      // B2 norms, B3 x, B4 h
    rmsnorm2: bg(weightsBuf, normsBuf, xBuf, hBuf, hBuf),
    q: bg(weightsBuf, scalesBuf, hBuf, qBuf, qBuf),            // B3 h, B4 q
    k: bg(weightsBuf, scalesBuf, hBuf, kBuf, kBuf),
    v: bg(weightsBuf, scalesBuf, hBuf, vBuf, vBuf),
    o: bg(weightsBuf, scalesBuf, attBuf, xBuf, xBuf),          // residual add into x
    gate: bg(weightsBuf, scalesBuf, hBuf, gateBuf, gateBuf),
    up: bg(weightsBuf, scalesBuf, hBuf, upBuf, upBuf),
    down: bg(weightsBuf, scalesBuf, actBuf, xBuf, xBuf),
    rope: bg(weightsBuf, ropeBuf, hBuf, qBuf, kBuf),           // B2 cos/sin, B4 q, B5 k
    copyK: bg(weightsBuf, scalesBuf, kBuf, kCache, kCache),    // B3 src, B4 dest
    copyV: bg(weightsBuf, scalesBuf, vBuf, vCache, vCache),
    scores: bg(weightsBuf, scalesBuf, qBuf, kCache, scoresBuf),// B3 q, B4 kcache, B5 scores
    softmax: bg(weightsBuf, scalesBuf, hBuf, scoresBuf, probsBuf), // B4 scores, B5 probs
    weighted: bg(weightsBuf, scalesBuf, vCache, attBuf, probsBuf), // B3 vcache, B4 att, B5 probs
    silu: bg(weightsBuf, scalesBuf, gateBuf, actBuf, upBuf),   // B3 gate, B4 act, B5 up
  };

  const layers = [];
  for (let i = 0; i < L; i++) {
    const pre = `layers.${i}.`;
    const w = (name) => {
      const t = tensors.get(name);
      return { w: t.offset / 4, s: scaleOff.get(name) };
    };
    layers.push({
      ln1: normOff.get(pre + "input_layernorm.weight"),
      ln2: normOff.get(pre + "post_attention_layernorm.weight"),
      q: w(pre + "self_attn.q_proj.weight"),
      k: w(pre + "self_attn.k_proj.weight"),
      v: w(pre + "self_attn.v_proj.weight"),
      o: w(pre + "self_attn.o_proj.weight"),
      gate: w(pre + "mlp.gate_proj.weight"),
      up: w(pre + "mlp.up_proj.weight"),
      down: w(pre + "mlp.down_proj.weight"),
      kvDim, inter,
      bg: shared,
    });
  }

  const stage = device.createBuffer({
    size: V * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });

  const embed = tensors.get("embed_tokens.weight");
  return {
    maxCtx, uniform, slots, p,
    kCache, vCache, logitsBuf, stage,
    sinOff: Math.floor(cos.byteLength / 4),
    embedByteOff: embed.offset,
    layers,
    bg: { embed: bgEmbed, logits: bgLogits, normOut: bgNormOut },
  };
}

function halfToFloat(h) {
  const s = (h & 0x8000) >> 15, e = (h & 0x7c00) >> 10, f = h & 0x03ff;
  if (e === 0) return (s ? -1 : 1) * Math.pow(2, -14) * (f / 1024);
  if (e === 0x1f) return f ? NaN : (s ? -Infinity : Infinity);
  return (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / 1024);
}
