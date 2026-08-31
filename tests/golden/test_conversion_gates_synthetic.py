#!/usr/bin/env python3
"""Synthetic end-to-end exercise of the three conversion STOP-GATE scripts
(REQ-CONV-2/3/4) — no /mnt assets, no fork build, no 147 GB anything.

Builds a minimal self-consistent pair (draft GGUF + model.safetensors, 81
tensors) plus a ref GGUF differing only in allowlisted metadata, then runs
each gate's main() in-process and asserts both PASS and FAIL modes:

  check_tensor_inventory: consistent pair passes; shape mismatch, wrong
    dflash.block_size, and a forbidden tensor (token_embd) each fail
  check_conv_base: torch-shape (2,2,4096) [side,tap,channel] written as-is
    (converter applies no transpose) passes content+layout+tap gates;
    tap-swapped data fails content
  diff_gguf_meta: allowlisted-only metadata diffs pass; a non-allowlisted
    key fails

Runs anywhere pip `gguf` exists (CI installs it); skips cleanly without.
"""

import importlib.util
import json
import os
import struct
import sys

import numpy as np
import pytest

GGUF_PY = os.environ.get("GGUF_PY", "")
try:
    if GGUF_PY:
        sys.path.insert(0, GGUF_PY)
    import gguf
except ImportError:
    gguf = None

# skipif (collected, skipped) — NOT pytest.skip(allow_module_level=True):
# an empty collection makes pytest exit 5 ("no tests collected"), which the
# gate runner rightly reports as a crash. This repo was bitten by exactly
# that before (release-v001 T3: "zero pytest functions, exit 5").
pytestmark = pytest.mark.skipif(
    gguf is None, reason="gguf package not importable (pip install gguf, or set GGUF_PY)")

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.normpath(os.path.join(HERE, "..", "..", "scripts"))
N_EMBD = 4096          # conv-base shape is asserted exactly: ne [4096, 2, 2]
N_LAYERS = 5
SEED = 123


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(SCRIPTS, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Shapes: real where a gate asserts them (conv bases), self-consistent-but-
# small everywhere else (the gates compare ckpt<->gguf, not against reality).
SHAPES = {
    "input_layernorm.weight": (256,),
    "post_attention_layernorm.weight": (256,),
    "self_attn.q_proj.weight": (256, 256),
    "self_attn.k_proj.weight": (64, 256),
    "self_attn.v_proj.weight": (64, 256),
    "self_attn.o_proj.weight": (256, 256),
    "self_attn.q_norm.weight": (128,),
    "self_attn.k_norm.weight": (128,),
    "mlp.gate_proj.weight": (512, 256),
    "mlp.up_proj.weight": (512, 256),
    "mlp.down_proj.weight": (256, 512),
    "attention_conv.base_kernel": (2, 2, N_EMBD),
    "attention_conv.kernel_projection.weight": (256, 512),
    "mlp_conv.base_kernel": (2, 2, N_EMBD),
    "mlp_conv.kernel_projection.weight": (256, 512),
}
TOP_SHAPES = {
    "norm.weight": (256,),
    "fc.weight": (128, 512),
    "hidden_norm.weight": (256,),
    "candidate_selector.hidden_projection.weight": (16, 64),
    "candidate_selector.predecessor_codebook": (256, 16),
    "candidate_selector.successor_codebook": (256, 16),
}


def build_ckpt(tap_swap=False):
    """81-tensor checkpoint dict; conv bases with tap-0 ~1.0, tap-1 ~0."""
    rng = np.random.default_rng(SEED)
    out = {}
    for bid in range(N_LAYERS):
        for suffix, shape in SHAPES.items():
            a = rng.normal(0, 0.02, size=shape).astype(np.float32)
            if suffix.endswith("conv.base_kernel"):
                a = np.zeros(shape, dtype=np.float32)
                a[:, 0, :] = 1.0 + 0.01 * rng.normal(size=shape[2]).astype(np.float32)
                a[:, 1, :] = 0.01 * rng.normal(size=shape[2]).astype(np.float32)
                if tap_swap:
                    a = np.ascontiguousarray(a[:, ::-1, :])
            out[f"layers.{bid}.{suffix}"] = a
    for name, shape in TOP_SHAPES.items():
        out[name] = rng.normal(0, 0.02, size=shape).astype(np.float32)
    return out


def write_safetensors(path, tensors):
    header, blob, off = {}, b"", 0
    for name, a in tensors.items():
        raw = a.astype(np.float32).tobytes()
        header[name] = {"dtype": "F32", "shape": list(a.shape),
                        "data_offsets": [off, off + len(raw)]}
        blob += raw
        off += len(raw)
    hj = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hj)))
        f.write(hj)
        f.write(blob)


MUST_MATCH_KVS = [
    ("dflash.block_count", 5), ("dflash.block_size", 8),
    ("dflash.conv_kernel_size", 2), ("dflash.conv_group_size", 16),
    ("dflash.selector_rank", 256), ("dflash.selector_top_k", 16),
    ("dflash.attention.head_count", 32), ("dflash.attention.head_count_kv", 8),
    ("dflash.attention.key_length", 128), ("dflash.attention.value_length", 128),
    ("dflash.attention.causal", False), ("dflash.attention.sliding_window", 2048),
    ("dflash.sliding_window_pattern", [1, 1, 1, 1, 1]),
    ("dflash.target_layers", [6, 15, 25, 34, 43]),
]
OURS_KVS = MUST_MATCH_KVS + [
    ("general.name", "dflash2-glm-synth"), ("general.license", "cc-by-nc-nd-4.0"),
    ("dflash.context_length", 1048576), ("dflash.embedding_length", 4096),
    ("dflash.feed_forward_length", 12288), ("dflash.rope.freq_base", 10000.0),
    ("dflash.attention.layer_norm_rms_epsilon", 1e-5),
]
REF_KVS = MUST_MATCH_KVS + [
    # allowlisted model-specific differences (GLM vs Qwen reference)
    ("general.name", "qwen-ref-synth"), ("general.license", "apache-2.0"),
    ("general.author", "z-lab"),
    ("dflash.context_length", 262144), ("dflash.embedding_length", 5120),
    ("dflash.feed_forward_length", 17408), ("dflash.rope.freq_base", 1e7),
    ("dflash.attention.layer_norm_rms_epsilon", 1e-6),
]


def write_gguf(path, kvs, tensors):
    w = gguf.GGUFWriter(path, arch="dflash")
    for key, val in kvs:
        if isinstance(val, bool):
            w.add_bool(key, val)
        elif isinstance(val, list):
            w.add_array(key, val)
        elif isinstance(val, str):
            w.add_string(key, val)
        elif isinstance(val, float):
            w.add_float32(key, val)
        else:
            w.add_uint32(key, val)
    for name, a in tensors.items():
        w.add_tensor(name, a)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


@pytest.fixture(scope="session")
def synth(tmp_path_factory):
    d = tmp_path_factory.mktemp("synth_gguf")
    inv = _load_script("check_tensor_inventory")

    def gguf_named(tensors):
        """Stand-in for the converter's rename table: checkpoint names ->
        GGUF names via the gate's own mapping (the gate then independently
        verifies shapes 1:1 through the dim reversal)."""
        out = {}
        for name, a in tensors.items():
            gname = inv.expected_gguf_name(name)
            assert gname is not None, f"unmapped fixture tensor: {name}"
            out[gname] = a
        return out

    ckpt = build_ckpt()
    ckpt_path = d / "model.safetensors"
    write_safetensors(ckpt_path, ckpt)
    named = gguf_named(ckpt)
    paths = {"ckpt": ckpt_path, "ours": d / "ours.gguf", "ref": d / "ref.gguf"}
    write_gguf(paths["ours"], OURS_KVS, named)
    write_gguf(paths["ref"], REF_KVS, named)

    shape_bad = dict(named)
    shape_bad["blk.2.attn_q.weight"] = np.zeros((128, 256), np.float32)
    paths["shape_bad"] = d / "shape_bad.gguf"
    write_gguf(paths["shape_bad"], OURS_KVS, shape_bad)

    paths["blocksize_bad"] = d / "blocksize_bad.gguf"
    write_gguf(paths["blocksize_bad"],
               [(k, 16 if k == "dflash.block_size" else v) for k, v in OURS_KVS], named)

    forbidden = dict(named)
    forbidden["token_embd.weight"] = np.zeros((256,), np.float32)
    paths["forbidden"] = d / "forbidden.gguf"
    write_gguf(paths["forbidden"], OURS_KVS, forbidden)

    paths["tapswap"] = d / "tapswap.gguf"
    write_gguf(paths["tapswap"], OURS_KVS, gguf_named(build_ckpt(tap_swap=True)))

    paths["extra_kv"] = d / "extra_kv.gguf"
    write_gguf(paths["extra_kv"], OURS_KVS + [("dflash.synthetic_only", 1)], named)
    return paths


@pytest.fixture(scope="session")
def gates():
    return {name: _load_script(name) for name in
            ("check_tensor_inventory", "check_conv_base", "diff_gguf_meta")}


def run_gate(monkeypatch, mod, *args):
    with monkeypatch.context() as m:
        m.setattr(sys, "argv", [f"{mod.__name__}.py", *map(str, args)])
        return mod.main()


def test_inventory_pass(monkeypatch, synth, gates):
    assert run_gate(monkeypatch, gates["check_tensor_inventory"],
                    synth["ours"], "--ckpt", synth["ckpt"]) == 0


def test_inventory_shape_mismatch(monkeypatch, synth, gates):
    assert run_gate(monkeypatch, gates["check_tensor_inventory"],
                    synth["shape_bad"], "--ckpt", synth["ckpt"]) == 1


def test_inventory_block_size_gate(monkeypatch, synth, gates):
    assert run_gate(monkeypatch, gates["check_tensor_inventory"],
                    synth["blocksize_bad"], "--ckpt", synth["ckpt"]) == 1


def test_inventory_forbidden_tensor(monkeypatch, synth, gates):
    assert run_gate(monkeypatch, gates["check_tensor_inventory"],
                    synth["forbidden"], "--ckpt", synth["ckpt"]) == 1


def test_conv_base_pass(monkeypatch, synth, gates):
    assert run_gate(monkeypatch, gates["check_conv_base"],
                    synth["ours"], "--ckpt", synth["ckpt"]) == 0


def test_conv_base_content_mismatch(monkeypatch, synth, gates):
    assert run_gate(monkeypatch, gates["check_conv_base"],
                    synth["tapswap"], "--ckpt", synth["ckpt"]) == 1


def test_diff_allowlisted_only(monkeypatch, synth, gates):
    assert run_gate(monkeypatch, gates["diff_gguf_meta"],
                    synth["ours"], synth["ref"]) == 0


def test_diff_unexpected_kv(monkeypatch, synth, gates):
    assert run_gate(monkeypatch, gates["diff_gguf_meta"],
                    synth["extra_kv"], synth["ref"]) == 1
