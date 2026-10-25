/* Fast incremental decoder for the exported single-layer character LSTM.
 * It carries the LSTM state between generated characters instead of rerunning
 * the same 128-character sequence for every next-character prediction.
 */
const sigmoid = (value) => value >= 0 ? 1 / (1 + Math.exp(-value)) : Math.exp(value) / (1 + Math.exp(value));

export function attachFastGeneration(model) {
  if (model.info?.name !== "v0idgpt-3-3M-reasoning" || model.info?.task !== "language") return model;

  const weights = model.getWeights();
  const embedding = weights["embedding_3m.W"];
  const inputWeights = weights["lstm_3m.Wx"];
  const recurrentWeights = weights["lstm_3m.Wh"];
  const recurrentBias = weights["lstm_3m.b"];
  const outputWeights = weights["logits_3m.W"];
  const outputBias = weights["logits_3m.b"];
  if (![embedding, inputWeights, recurrentWeights, recurrentBias, outputWeights, outputBias].every(Boolean)) {
    throw new Error("Fast decoder could not find the 3M model's expected weight tensors");
  }

  const embedSize = 128;
  const hiddenSize = 800;
  const vocabSize = model.info.tokenizer.vocabSize;
  const gateSize = hiddenSize * 4;
  const seqLen = model.info.tokenizer.seqLen;
  // ID 0 is the exported model's <pad> token; its printable spelling is not
  // itself a character token sequence.
  const actualPadId = 0;
  const padEmbeddingIsZero = embedding.slice(actualPadId * embedSize, (actualPadId + 1) * embedSize).every((x) => x === 0);
  const biasIsZero = recurrentBias.every((x) => x === 0);
  const rngForSeed = (seed) => ({
    state: (Number(seed) >>> 0) || 1,
    next() {
      this.state = (this.state + 0x6d2b79f5) >>> 0;
      let value = this.state;
      value = Math.imul(value ^ (value >>> 15), value | 1);
      value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
      return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
    },
  });
  const sample = (probs, temperature, topK, rng) => {
    let best = 0;
    if (!Number.isFinite(temperature) || temperature <= 0.01) {
      for (let i = 1; i < probs.length; i++) if (probs[i] > probs[best]) best = i;
      return best;
    }
    const invT = 1 / temperature;
    const logits = Array.from(probs, (p) => Math.log(Math.max(Number(p), 1e-9)) * invT);
    const k = Math.max(1, Math.min(Math.floor(topK) || probs.length, probs.length));
    const candidates = Array.from({ length: probs.length }, (_, i) => i)
      .sort((a, b) => logits[b] - logits[a])
      .slice(0, k);
    let max = -Infinity;
    for (const id of candidates) if (logits[id] > max) max = logits[id];
    const masses = candidates.map((id) => Math.exp(logits[id] - max));
    const total = masses.reduce((sum, mass) => sum + mass, 0);
    let draw = rng.next() * total;
    for (let i = 0; i < candidates.length; i++) {
      draw -= masses[i];
      if (draw <= 0) return candidates[i];
    }
    return candidates[candidates.length - 1] ?? best;
  };

  const fast = {
    createGenerationSession(prompt) {
      const hidden = new Float32Array(hiddenSize);
      const cell = new Float32Array(hiddenSize);
      const gatePre = new Float64Array(gateSize);
      const ids = model.tokenize(prompt).slice(-seqLen);

      const consume = (tokenId) => {
        const embeddingOffset = tokenId * embedSize;
        for (let j = 0; j < gateSize; j++) gatePre[j] = recurrentBias[j];

        // Matrix-vector products are accumulated in input-major order so each
        // weight row is contiguous in memory. For each gate, additions still
        // occur in the same feature order as the standalone runtime.
        for (let i = 0; i < embedSize; i++) {
          const value = embedding[embeddingOffset + i];
          if (value === 0) continue;
          const offset = i * gateSize;
          for (let j = 0; j < gateSize; j++) gatePre[j] += value * inputWeights[offset + j];
        }
        for (let i = 0; i < hiddenSize; i++) {
          const value = hidden[i];
          if (value === 0) continue;
          const offset = i * gateSize;
          for (let j = 0; j < gateSize; j++) gatePre[j] += value * recurrentWeights[offset + j];
        }

        for (let u = 0; u < hiddenSize; u++) {
          const inputGate = sigmoid(gatePre[u]);
          const forgetGate = sigmoid(gatePre[hiddenSize + u]);
          const candidate = Math.tanh(gatePre[2 * hiddenSize + u]);
          const outputGate = sigmoid(gatePre[3 * hiddenSize + u]);
          cell[u] = forgetGate * cell[u] + inputGate * candidate;
          hidden[u] = outputGate * Math.tanh(cell[u]);
        }
      };

      // Match the standalone runtime's left-padded context. The trainer keeps
      // the pad embedding and recurrent biases at zero, so this loop is skipped
      // in the normal export; it remains correct if that ever changes.
      if (!(padEmbeddingIsZero && biasIsZero)) {
        for (let i = ids.length; i < seqLen; i++) consume(actualPadId);
      }
      for (const id of ids) consume(id);

      return {
        predict() {
          // Mirror the runtime's float32 dense accumulation and softmax.
          const logits = new Float32Array(vocabSize);
          for (let i = 0; i < hiddenSize; i++) {
            const value = hidden[i];
            const offset = i * vocabSize;
            for (let j = 0; j < vocabSize; j++) logits[j] += value * outputWeights[offset + j];
          }
          for (let j = 0; j < vocabSize; j++) logits[j] += outputBias[j];
          let max = -Infinity;
          for (let j = 0; j < vocabSize; j++) if (logits[j] > max) max = logits[j];
          const probs = new Float32Array(vocabSize);
          let sum = 0;
          for (let j = 0; j < vocabSize; j++) {
            probs[j] = Math.exp(logits[j] - max);
            sum += probs[j];
          }
          const inv = 1 / (sum || 1e-12);
          for (let j = 0; j < vocabSize; j++) probs[j] *= inv;
          return { raw: probs };
        },
        push(tokenId) {
          if (!Number.isInteger(tokenId) || tokenId < 0 || tokenId >= vocabSize) {
            throw new RangeError(`Invalid generated token ID: ${tokenId}`);
          }
          consume(tokenId);
        },
      };
    },
  };

  const generateDetailed = (prompt, options = {}) => {
    options = options || {};
    const maxTokens = options.maxTokens == null ? 60 : Math.max(0, Math.floor(options.maxTokens));
    const temperature = options.temperature == null ? model.info.sampling.temperature : Number(options.temperature);
    const topK = options.topK == null ? model.info.sampling.topK : Number(options.topK);
    const seed = options.seed == null ? Math.floor(Math.random() * 4294967296) : Number(options.seed);
    const eos = model.info.tokenizer.eos;
    const stopList = Array.isArray(options.stop)
      ? options.stop
      : options.stopAtEos !== false && eos ? [eos] : [];
    const stop = Object.create(null);
    for (const token of stopList) stop[token] = true;

    const promptIds = model.tokenize(prompt);
    if (promptIds.length === 0) throw new Error("The prompt contains no characters from the vocabulary");
    const rng = rngForSeed(seed);
    const session = fast.createGenerationSession(prompt);
    const ids = [];
    const tokens = [];
    let stoppedBy = null;
    for (let i = 0; i < maxTokens; i++) {
      const probs = session.predict().raw;
      const id = sample(probs, temperature, topK, rng);
      const token = model.detokenize([id]);
      if (token === undefined) throw new Error(`Model produced class id ${id} outside the vocabulary`);
      if (stop[token]) {
        stoppedBy = token;
        break;
      }
      ids.push(id);
      tokens.push(token);
      session.push(id);
    }
    const text = model.detokenize(ids);
    return {
      text,
      fullText: String(prompt) + text,
      ids,
      tokens,
      promptIds,
      stoppedBy,
      settings: { maxTokens, temperature, topK, seed, stop: stopList },
    };
  };

  Object.defineProperties(model, {
    createGenerationSession: {
      value: fast.createGenerationSession,
      enumerable: false,
      configurable: false,
      writable: false,
    },
    generateDetailed: {
      value: generateDetailed,
      enumerable: false,
      configurable: false,
      writable: false,
    },
    generate: {
      value(prompt, options) {
        const result = generateDetailed(prompt, options || {});
        return options && options.includePrompt ? result.fullText : result.text;
      },
      enumerable: false,
      configurable: false,
      writable: false,
    },
  });
  return model;
}
