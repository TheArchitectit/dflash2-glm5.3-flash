#!/usr/bin/env python3
"""Sprint 1.9 — conv base layout golden check (risk 3, REQ-CONV-4).

Usage:
    check_conv_base.py <draft.gguf> [--ckpt /path/to/model.safetensors]

For each of the 10 conv-base tensors (5 layers x attn+ffn):
  1. GGUF ne == [4096, 2, 2] == [n_embd, kernel, 2]
  2. Element semantics vs checkpoint: gguf[c][t][s] == ckpt[s][t][c]
     (checkpoint layout [side, tap, channel]; the torch->ggml dim reversal
     alone must land the right layout — verified against the reference GGUF)
  3. Tap-0 stats dominate tap-1 (SGLang inits base_kernel[:, 0] = 1.0,
     dflash.py:433-435: tap-0 mean should be O(1), tap-1 near 0)

Exit 0 = pass.
"""
import struct
import sys
from pathlib import Path

import numpy as np

CKPT_DEFAULT = "/mnt/ollama/models/glm-5.3-flash/dflash2/model.safetensors"


def read_safetensors(path):
    with open(path, "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        import json
        hdr = json.loads(f.read(hlen))
        out = {}
        for name, meta in hdr.items():
            if name == "__metadata__":
                continue
            dtype, shape, (start, end) = meta["dtype"], meta["shape"], meta["data_offsets"]
            f.seek(8 + hlen + start)
            raw = f.read(end - start)
            np_dtype = {"BF16": (np.dtype("uint16"), 2), "F16": (np.dtype("uint16"), 2),
                        "F32": (np.dtype("float32"), 4)}.get(dtype)
            if np_dtype is None:
                raise ValueError(f"unexpected dtype {dtype} for {name}")
            dt, _ = np_dtype
            arr = np.frombuffer(raw, dtype=dt)
            if dtype in ("BF16",):
                arr = arr.view(np.uint16).astype(np.uint32) << 16
                arr = arr.view(np.float32)
            out[name] = arr.reshape(shape)
        return out


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    gguf_path = Path(sys.argv[1])
    ckpt_path = sys.argv[sys.argv.index("--ckpt") + 1] if "--ckpt" in sys.argv else CKPT_DEFAULT

    sys.path.insert(0, "/mnt/ollama/models/llama-cpp-glm5/gguf-py")
    from gguf import GGUFReader  # noqa: E402

    ckpt = read_safetensors(ckpt_path)
    reader = GGUFReader(gguf_path)

    gguf_map = {}
    for t in reader.tensors:
        if "conv_base" in t.name:
            # t.shape is ne-order (fastest first); ggml data is row-major over
            # reversed(ne), so the numpy view must reshape with dims reversed.
            gguf_map[t.name] = t.data.reshape([int(x) for x in reversed(t.shape)])

    ok = True
    n_checked = 0
    for bid in range(5):
        for conv_type, ckpt_name in (("attn", f"layers.{bid}.attention_conv.base_kernel"),
                                     ("ffn", f"layers.{bid}.mlp_conv.base_kernel")):
            gguf_name = f"blk.{bid}.{conv_type}_conv_base"
            ck = ckpt[ckpt_name]  # [side, tap, channel] = [2,2,4096]
            if gguf_name not in gguf_map:
                print(f"MISSING {gguf_name} in GGUF")
                ok = False
                continue
            g = gguf_map[gguf_name]  # [side, tap, channel] = [2,2,4096] (matches ckpt)
            if list(g.shape) != [2, 2, 4096]:
                ok = False
                print(f"SHAPE FAIL {gguf_name}: {list(g.shape)} != [2,2,4096]")
                continue
            # semantic check: raw bytes must match the checkpoint exactly
            # (the torch->ggml dim reversal alone lands the correct layout —
            # verified byte-for-byte; no explicit transpose in the converter)
            if not np.array_equal(g, ck):
                maxdiff = float(np.max(np.abs(g.astype(np.float32) - ck.astype(np.float32))))
                ok = False
                print(f"CONTENT FAIL {gguf_name}: GGUF != checkpoint (max abs diff {maxdiff})")
            # tap stats: tap-0 ~1.0, tap-1 ~0
            t0 = float(np.mean(ck[:, 0, :]))
            t1 = float(np.mean(np.abs(ck[:, 1, :])))
            if not (0.1 < abs(t0) < 10.0) or t1 > 1.0:
                ok = False
                print(f"TAP STATS FAIL {gguf_name}: tap0 mean={t0:.4f} (want O(1)), tap1 |mean|={t1:.4f} (want ~0)")
            n_checked += 1

    if not ok:
        print("CONV BASE CHECK FAILED")
        return 1
    print(f"CONV BASE CHECK PASSED ({n_checked} tensors, layout + content + tap stats)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
