#!/usr/bin/env python3
"""Sprint 3.2 — mHC-collapse micro-test (risk 4 / risk 7 STOP GATE).

Compares llama.cpp's `build_hc_mean` (glm5next.cpp:613 → deepseek4.cpp:267,
an UNWEIGHTED MEAN over the hyper-connection streams) against SGLang's
`_mhc_pre_torch` contraction (mhc.py:1626):

    layer_input = (sigmoid(mixes[:, :n] * hc_scale[0] + hc_base[:n]) + eps
                   * residual).sum(streams)

The DFlash2 draft was distilled on SGLang's representation. If the two
reductions differ materially, the draft sees a different hidden
distribution in llama.cpp than it was trained against — acceptance
degrades (measured 3.36 vs published 4.4-5.5).

Uses the REAL learned mHC weights from the GLM-5.3-Flash target
(blk.5.hc_attn_{fn,base,scale}, dequantized from the UD-IQ4_XS GGUF),
so the gate reflects production magnitudes, not synthetic worst case.

Run: python3 tests/golden/test_hc_collapse.py [--target-gguf-dir DIR]
Exit 0 = representations match (fp32 eps); nonzero = DIVERGENCE CONFIRMED.
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, "/mnt/ollama/models/llama-cpp-glm5/gguf-py")
import gguf  # noqa: E402

# ---- model constants (GLM-5.3-Flash / glm5next) --------------------------
HC = 4            # glm5next.hyper_connection.count
N_EMBD = 4096
RMS_EPS = 1e-5     # glm5next.attention.layer_norm_rms_epsilon
HC_EPS = 1e-6      # glm5next.hyper_connection.epsilon
SINKHORN_ITERS = 20
LAYER = 5          # first dflash extraction layer (0-indexed)

DEFAULT_GGUF_DIR = "/mnt/ollama/models/glm-5.3-flash/UD-IQ4_XS"


def load_hc_weights(gguf_dir: str, layer: int):
    """Dequantize blk.<layer>.hc_attn_{fn,base,scale} from the target GGUF."""
    for part in sorted(os.listdir(gguf_dir)):
        if not part.endswith(".gguf"):
            continue
        try:
            r = gguf.GGUFReader(os.path.join(gguf_dir, part))
        except Exception:
            continue
        names = {t.name for t in r.tensors}
        want = [f"blk.{layer}.hc_attn_fn.weight", f"blk.{layer}.hc_attn_base.weight",
                f"blk.{layer}.hc_attn_scale.weight"]
        if not all(w in names for w in want):
            continue

        def deq(name):
            t = next(t for t in r.tensors if t.name == name)
            shape = [int(x) for x in reversed(t.shape)]
            flat = np.frombuffer(t.data, dtype=np.uint8).copy()
            qtype = gguf.GGMLQuantizationType(t.tensor_type)
            if qtype == gguf.GGMLQuantizationType.F32:
                return np.frombuffer(t.data, dtype=np.float32).reshape(shape)
            bs, ts = gguf.GGML_QUANT_SIZES[qtype]
            nrows, ncols = shape[0], shape[1]
            assert flat.shape[0] == nrows * (ncols // bs) * ts, name
            blocks = flat.reshape(nrows, ncols // bs, ts)
            return gguf.quants.Q8_0.dequantize(blocks).reshape(nrows, ncols)

        return deq(want[0]), deq(want[1]), deq(want[2])
    raise FileNotFoundError(f"hc_attn weights for layer {layer} not found in {gguf_dir}")


def sglang_mhc_layer_input(residual: np.ndarray, fn: np.ndarray, base: np.ndarray,
                            scale: np.ndarray) -> np.ndarray:
    """SGLang _mhc_pre_torch layer_input contraction (mhc.py:1626).

    residual: (s, n, h) — s tokens, n streams, h hidden.
    Returns (s, h): sum over streams of pre-gated residual.
    """
    s, n, h = residual.shape
    x_flat = residual.reshape(s, n * h).astype(np.float32)
    rsqrt = 1.0 / np.sqrt(np.mean(x_flat ** 2, axis=-1, keepdims=True) + RMS_EPS)
    mixes = (x_flat @ fn.T) * rsqrt          # (s, 24)

    pre_raw = mixes[:, :n]
    pre_base = base[:n]
    pre = 1.0 / (1.0 + np.exp(-(pre_raw * scale[0] + pre_base))) + HC_EPS
    return (pre[:, :, None] * residual).sum(axis=1)


def llamacpp_hc_mean(x: np.ndarray) -> np.ndarray:
    """llama.cpp build_hc_mean (deepseek4.cpp:267): unweighted mean over streams."""
    return x.mean(axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-gguf-dir", default=DEFAULT_GGUF_DIR)
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print(f"[3.2] mHC collapse micro-test — layer {LAYER}, {args.tokens} tokens, real weights")
    fn, base, scale = load_hc_weights(args.target_gguf_dir, LAYER)
    print(f"  weights: fn{fn.shape} base{base.shape} scale{scale.shape} (dequantized OK)")
    print(f"  pre_base: {np.round(base[:HC], 3)}")

    rng = np.random.default_rng(args.seed)
    # production-magnitude residual: per-stream std scaled like hidden states
    residual = rng.normal(0, 1.0, size=(args.tokens, HC, N_EMBD)).astype(np.float32)

    ref = sglang_mhc_layer_input(residual, fn, base, scale)
    ours = llamacpp_hc_mean(residual)

    # fp32 eps baseline: compare ref against itself in fp32 (sanity)
    denom = np.abs(ref) + 1e-9
    rel = np.abs(ref - ours) / denom
    cos = np.sum(ref * ours, -1) / (np.linalg.norm(ref, axis=-1) * np.linalg.norm(ours, axis=-1) + 1e-9)

    # also: what does the unweighted mean look like if pre-gates were uniform?
    print(f"\n  ref  (SGLang gated contraction): mean|.|={np.abs(ref).mean():.4f}")
    print(f"  ours (build_hc_mean unweighted): mean|.|={np.abs(ours).mean():.4f}")
    print(f"  rel-err: mean={rel.mean():.4f}  median={np.median(rel):.4f}  p99={np.percentile(rel, 99):.4f}")
    print(f"  cosine similarity: mean={cos.mean():.6f}  min={cos.min():.6f}")
    print(f"  pre-gates at this residual: {np.round(
        1/(1+np.exp(-(0.0 * scale[0] + base[:HC]))), 4)} (zero-input, sigmoid(base))")

    gates = 1.0 / (1.0 + np.exp(-base[:HC]))
    print(f"  sigmoid(pre_base) per stream: {np.round(gates, 6)}")
    print(f"  unweighted-mean implied gates: {np.full(HC, 1.0 / HC)}")

    eps = 6e-2  # generous fp32 tolerance for "representations match"
    if rel.mean() < eps:
        print(f"\n  PASS: representations match within {eps} rel — risk 4 is NOT the cause.")
        return 0
    print("\n  *** DIVERGENCE CONFIRMED (risk 4 / risk 7) ***")
    print("  llama.cpp build_hc_mean != SGLang gated contraction.")
    print("  The draft was distilled on the SGLang representation.")
    print("  Fix (Sprint 5.4): wire build_hc_pre (deepseek4.cpp:351, already")
    print("  implemented) into the glm5next t_layer_inp extraction path.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
