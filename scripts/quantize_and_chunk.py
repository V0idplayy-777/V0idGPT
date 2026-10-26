#!/usr/bin/env python3
"""Quantize a 109M bundle.mjs to 8-bit and split into <100MB chunks for GitHub push."""
import json, base64, re, sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
MODEL_DIR=ROOT/"models/v0idgpt-4-109M"
BUNDLE=MODEL_DIR/"bundle.mjs"
OUT_DIR=MODEL_DIR

def load_bundle():
    txt=BUNDLE.read_text()
    # Extract JSON after 'export default '
    m=re.search(r'export default\s+(\{.*\});\s*$', txt, re.DOTALL)
    if not m:
        # Try without trailing ;
        m=re.search(r'export default\s+(\{.*\})', txt, re.DOTALL)
    if not m:
        raise ValueError("Could not find bundle JSON")
    return json.loads(m.group(1))

def quantize_tensor(name, entry):
    shape=entry["shape"]
    raw_b64=entry["data"]
    raw=base64.b64decode(raw_b64)
    arr=np.frombuffer(raw, dtype=np.float32).reshape(shape) if shape else np.frombuffer(raw, dtype=np.float32)
    # Compute per-tensor min/max
    mn=float(arr.min())
    mx=float(arr.max())
    if mx==mn:
        scale=1.0
        q=np.zeros(arr.size, dtype=np.uint8)
    else:
        scale=(mx-mn)/255.0
        q=np.clip(np.round((arr - mn)/scale), 0, 255).astype(np.uint8)
    return {
        "shape": shape,
        "dtype": "uint8",
        "min": mn,
        "scale": scale,
        "checksum": entry.get("checksum",""),
        "quantized": True,
        "originalDtype": "float32",
        # Don't store data inline, will be stored in binary chunks
        "dataRef": name,
        "size": int(q.size)
    }, q.tobytes()

def main():
    print("Loading bundle...")
    bundle=load_bundle()
    weights=bundle["weights"]
    print(f"Original weights: {len(weights)} tensors")
    total_params=sum(np.prod(v["shape"]) if v["shape"] else 1 for v in weights.values())
    print(f"Total params {total_params/1e6:.1f}M")
    # Quantize each tensor
    quantized_meta={}
    blobs=[]
    total_bytes=0
    for k in sorted(weights.keys()):
        meta, blob = quantize_tensor(k, weights[k])
        quantized_meta[k]=meta
        blobs.append((k, blob))
        total_bytes+=len(blob)
    print(f"Quantized total {total_bytes/1e6:.1f} MB (vs {total_params*4/1e6:.1f} MB float32)")
    # Build new bundle with quantized metadata (no inline data)
    bundle["weights"]=quantized_meta
    bundle["quantization"]={"method":"per-tensor uint8 linear","originalBytes":int(total_params*4),"quantizedBytes":total_bytes}
    bundle["provenance"]["quantized"]=True
    # Estimate JSON size
    manifest_json=json.dumps(bundle, separators=(",",":"))
    print(f"Manifest JSON {len(manifest_json)/1e6:.2f} MB")
    # Split blobs into chunks <90MB each (GitHub limit 100MB, stay safe)
    CHUNK_SIZE=90*1024*1024
    chunks=[]
    cur=b''
    cur_names=[]
    for name, blob in blobs:
        if len(cur)+len(blob) > CHUNK_SIZE and cur:
            chunks.append((cur, cur_names))
            cur=b''
            cur_names=[]
        cur+=blob
        cur_names.append(name)
    if cur:
        chunks.append((cur, cur_names))
    print(f"Split into {len(chunks)} chunks")
    # Write manifest and chunks
    # Manifest will be bundle.quant.mjs with metadata but no inline data
    manifest_text="// v0idgpt-4-109M quantized manifest\n export default "+manifest_json+";\n"
    (OUT_DIR/"bundle.quant.mjs").write_text(manifest_text, encoding="utf-8")
    print(f"Wrote bundle.quant.mjs {len(manifest_text)/1e6:.2f} MB")
    # Write chunks as binary files
    offset=0
    for idx, (data, names) in enumerate(chunks):
        path=OUT_DIR/f"weights-{idx:02d}.bin"
        path.write_bytes(data)
        print(f"Wrote {path} {len(data)/1e6:.1f} MB ({len(names)} tensors: {', '.join(names[:3])}...)")
        offset+=len(data)
    # Also write index file for loader
    index={
        "chunks": [{"file": f"weights-{i:02d}.bin", "size": len(c[0]), "tensors": c[1]} for i,c in enumerate(chunks)],
        "quantization": bundle["quantization"],
        "totalParams": int(total_params)
    }
    (OUT_DIR/"weights-index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    print("Done. Loader should fetch bundle.quant.mjs + weights-*.bin and dequantize in JS.")
    # Also estimate base64 size if we had stored inline (for comparison)
    b64_size=total_bytes*4/3/1e6
    print(f"If stored as base64 inline, would be {b64_size:.1f} MB + manifest")

if __name__=="__main__":
    main()
