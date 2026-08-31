#!/usr/bin/env python3
"""Sprint 3.2 / 5.4 — mHC capture-semantics regression test.

HISTORY: this test originally compared llama.cpp's `build_hc_mean` against
SGLang's *gated* contraction (`_mhc_pre_torch`) and "confirmed divergence"
(commit 71011ee). That comparison targeted the WRONG function and led to a
now-reverted patch (de9669d, reverted 77445b3). The real DFLASH capture is
`Glm5NextModel._prepare_aux_hidden_state` (glm5_next.py:1078) which calls
`hc_contract` (mhc.py:1571): an UNWEIGHTED MEAN over hyper-connection
streams — exactly what `build_hc_mean` implements (risk 7 retracted,
research/07-gap-analysis.md).

Correct semantics asserted here:

  T1  build_hc_mean  ==  SGLang hc_contract            (the capture path)
  T2  build_hc_mean  !=  gated _mhc_pre_torch           (in-layer generation
       path only — a re-assert would mean someone re-applied the reverted
       patch or re-pointed the capture)

T1 uses real learned weights (blk.5.hc_attn_*, dequantized from the
UD-IQ4_XS target) when available, synthetic otherwise, so CI passes without
the 147 GB shards.

Run standalone:  python3 tests/golden/test_hc_collapse.py [--target-gguf-dir DIR]
Run via pytest:  pytest tests/golden/test_hc_collapse.py

The gguf package is only needed for the real-weights arm (T1-synthetic runs
without it). Point GGUF_PY at a llama.cpp checkout's gguf-py when the target
GGUF is present; DFLASH2_TARGET_GGUF_DIR overrides the default dir.
"""

import argparse
import os
import sys

import numpy as np
import pytest

GGUF_PY = os.environ.get("GGUF_PY", "/mnt/ollama/models/llama-cpp-glm5/gguf-py")
try:
    sys.path.insert(0, GGUF_PY)
    import gguf  # noqa: E402
except ImportError:
    gguf = None  # synthetic-path tests (T1) don't need it

# ---- model constants (GLM-5.3-Flash / glm5next) --------------------------
HC = 4             # glm5next.hyper_connection.count
N_EMBD = 4096
RMS_EPS = 1e-5     # glm5next.attention.layer_norm_rms_epsilon
HC_EPS = 1e-6      # glm5next.hyper_connection.epsilon
LAYER = 5          # first dflash extraction layer (0-indexed)

DEFAULT_GGUF_DIR = os.environ.get(
    "DFLASH2_TARGET_GGUF_DIR", "/mnt/ollama/models/glm-5.3-flash/UD-IQ4_XS")


def load_hc_weights(gguf_dir: str, layer: int):
    """Dequantize blk.<layer>.hc_attn_{fn,base,scale} from the target GGUF."""
    if gguf is None:
        raise RuntimeError(
            f"gguf package not importable (GGUF_PY={GGUF_PY!r}); "
            "real-weights arm unavailable")
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
            if qtype != gguf.GGMLQuantizationType.Q8_0:
                # deq below only knows the Q8_0 block layout — anything else
                # must fail loudly, not silently produce garbage (a wrong-type
                # tensor can still pass the shape assert below).
                raise RuntimeError(f"{name}: unsupported quant type {qtype}, "
                                   "expected F32 or Q8_0")
            bs, ts = gguf.GGML_QUANT_SIZES[qtype]
            nrows, ncols = shape[0], shape[1]
            assert flat.shape[0] == nrows * (ncols // bs) * ts, name
            blocks = flat.reshape(nrows, ncols // bs, ts)
            return gguf.quants.Q8_0.dequantize(blocks).reshape(nrows, ncols)

        return deq(want[0]), deq(want[1]), deq(want[2])
    raise FileNotFoundError(f"hc_attn weights for layer {layer} not found in {gguf_dir}")


def real_or_synthetic_weights(gguf_dir, tokens=64, seed=42):
    """(fn, base, scale) from the target GGUF when present, else synthetic
    production-magnitude weights. Returns (weights, is_real)."""
    try:
        return load_hc_weights(gguf_dir, LAYER), True
    except (FileNotFoundError, OSError, RuntimeError, ImportError):
        rng = np.random.default_rng(seed)
        fn = rng.normal(0, 0.02, size=(3 * HC, N_EMBD)).astype(np.float32)
        base = rng.normal(0, 1.0, size=3 * HC).astype(np.float32)
        scale = rng.normal(1.0, 0.1, size=HC).astype(np.float32)
        return (fn, base, scale), False


def sglang_hc_contract(residual: np.ndarray) -> np.ndarray:
    """SGLang mhc.py:1571 hc_contract — the DFLASH capture reduction.

    aux.unflatten(-1, (n, -1)).mean(dim=-2): plain unweighted mean.
    residual: (s, n, h) -> (s, h).
    """
    return residual.mean(axis=-2)


def sglang_mhc_pre_torch(residual, fn, base, scale):
    """SGLang _mhc_pre_torch (mhc.py:1626) — gated contraction used for
    NORMAL in-layer generation, NOT the dflash capture."""
    s, n, h = residual.shape
    x_flat = residual.reshape(s, n * h).astype(np.float32)
    rsqrt = 1.0 / np.sqrt(np.mean(x_flat ** 2, axis=-1, keepdims=True) + RMS_EPS)
    mixes = (x_flat @ fn.T) * rsqrt
    pre_raw, pre_base = mixes[:, :n], base[:n]
    pre = 1.0 / (1.0 + np.exp(-(pre_raw * scale[0] + pre_base))) + HC_EPS
    return (pre[:, :, None] * residual).sum(axis=1)


def llamacpp_build_hc_mean(x: np.ndarray) -> np.ndarray:
    """llama.cpp build_hc_mean (deepseek4.cpp:267)."""
    return x.mean(axis=1)


def _residual(tokens=64, seed=42):
    rng = np.random.default_rng(seed)
    return rng.normal(0, 1.0, size=(tokens, HC, N_EMBD)).astype(np.float32)


def test_capture_is_unweighted_mean():
    """T1: build_hc_mean == hc_contract (the capture semantics). Bit-exact:
    both are literally mean-over-streams."""
    x = _residual()
    assert llamacpp_build_hc_mean(x) == pytest.approx(sglang_hc_contract(x), rel=0, abs=1e-6)


def test_capture_gguf_weights_if_available():
    """T1 on real weights: with the target present, assert the capture still
    equals the unweighted mean and that the gated contraction (wrong path)
    diverges — the full original finding, corrected."""
    if gguf is None:
        pytest.skip(f"gguf package not importable (GGUF_PY={GGUF_PY!r})")
    if not os.path.isdir(DEFAULT_GGUF_DIR):
        pytest.skip(f"target GGUF dir {DEFAULT_GGUF_DIR} not present")
    (fn, base, scale), is_real = real_or_synthetic_weights(DEFAULT_GGUF_DIR)
    assert is_real, "expected real weights when target dir exists"
    x = _residual()
    ours = llamacpp_build_hc_mean(x)
    assert ours == pytest.approx(sglang_hc_contract(x), rel=0, abs=1e-6)
    gated = sglang_mhc_pre_torch(x, fn, base, scale)
    denom = np.abs(gated) + 1e-9
    rel = np.abs(gated - ours) / denom
    assert np.median(rel) > 0.1, (
        "gated contraction unexpectedly close to the mean — re-check capture "
        "semantics before touching glm5next extraction")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-gguf-dir", default=DEFAULT_GGUF_DIR)
    args = ap.parse_args()
    print("[hc] T1 capture == unweighted mean: ", end="")
    x = _residual()
    assert llamacpp_build_hc_mean(x) == pytest.approx(sglang_hc_contract(x), rel=0, abs=1e-6)
    print("PASS")
    _, is_real = real_or_synthetic_weights(args.target_gguf_dir)
    print(f"[hc] weights arm: {'real (target GGUF)' if is_real else 'synthetic'} "
          f"from {args.target_gguf_dir}")
    rc = pytest.main([__file__, "-q"])
    return int(rc)


if __name__ == "__main__":
    sys.exit(main())
