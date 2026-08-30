#!/usr/bin/env python3
"""Sprint 1.3 / 1.6 / 1.10 — tensor inventory + metadata check for the DFlash2 GGUF.

Usage:
    check_tensor_inventory.py <draft.gguf> [--ckpt /path/to/model.safetensors]

Compares the converted GGUF against the checkpoint's safetensors header:
  - 1:1 tensor name + shape match through the torch->ggml dim reversal
    (torch [a,b,c] -> ggml ne [c,b,a])
  - no token_embd / output / d2t (headless draft)
  - dflash.* metadata asserts (block_size=8, layers [6,15,25,34,43], ...)

Exit 0 = pass. Any mismatch prints the diff and exits 1 (STOP GATE).
"""
import json
import os
import struct
import sys
from pathlib import Path

# torch suffix -> gguf suffix (per-layer). Convention verified against the
# reference GGUF (incoai/Qwen3.8-27B-DFlash2-GGUF): all per-layer weights carry
# .weight except the conv bases; o_proj maps to attn_output; codebooks have no
# suffix.
PER_LAYER = {
    "input_layernorm.weight":                       "attn_norm.weight",
    "post_attention_layernorm.weight":              "ffn_norm.weight",
    "self_attn.q_proj.weight":                      "attn_q.weight",
    "self_attn.k_proj.weight":                     "attn_k.weight",
    "self_attn.v_proj.weight":                     "attn_v.weight",
    "self_attn.o_proj.weight":                      "attn_output.weight",
    "self_attn.q_norm.weight":                      "attn_q_norm.weight",
    "self_attn.k_norm.weight":                      "attn_k_norm.weight",
    "mlp.gate_proj.weight":                         "ffn_gate.weight",
    "mlp.up_proj.weight":                           "ffn_up.weight",
    "mlp.down_proj.weight":                         "ffn_down.weight",
    "attention_conv.base_kernel":                   "attn_conv_base",
    "attention_conv.kernel_projection.weight":      "attn_conv_proj.weight",
    "mlp_conv.base_kernel":                         "ffn_conv_base",
    "mlp_conv.kernel_projection.weight":            "ffn_conv_proj.weight",
}

TOP_LEVEL = {
    "norm.weight":                                  "output_norm.weight",
    "fc.weight":                                    "fc.weight",
    "hidden_norm.weight":                           "enc.output_norm.weight",
    "candidate_selector.hidden_projection.weight":  "selector_hidden.weight",
    "candidate_selector.predecessor_codebook":     "selector_predecessor.weight",
    "candidate_selector.successor_codebook":        "selector_successor.weight",
}


def read_safetensors_header(path: str) -> dict[str, list[int]]:
    with open(path, "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(hlen))
    return {k: v["shape"] for k, v in hdr.items() if k != "__metadata__"}


def expected_gguf_name(ckpt_name: str) -> str | None:
    if ckpt_name.startswith("layers."):
        rest = ckpt_name[len("layers."):]
        bid, _, suffix = rest.partition(".")
        gguf_suffix = PER_LAYER.get(suffix)
        if gguf_suffix is None:
            return None
        return f"blk.{int(bid)}.{gguf_suffix}"
    return TOP_LEVEL.get(ckpt_name)


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    gguf_path = Path(sys.argv[1])
    ckpt = sys.argv[sys.argv.index("--ckpt") + 1] if "--ckpt" in sys.argv else \
        os.environ.get("DFLASH2_CKPT", "/mnt/ollama/models/glm-5.3-flash/dflash2/model.safetensors")

    sys.path.insert(0, os.environ.get("GGUF_PY", "/mnt/ollama/models/llama-cpp-glm5/gguf-py"))
    from gguf import GGUFReader  # noqa: E402

    ckpt_shapes = read_safetensors_header(ckpt)
    reader = GGUFReader(gguf_path)

    gguf_tensors: dict[str, list[int]] = {}
    for t in reader.tensors:
        gguf_tensors[t.name] = [int(n) for n in t.shape]

    # build expected: {gguf_name: ne list}
    expected: dict[str, list[int]] = {}
    for ckpt_name, torch_shape in ckpt_shapes.items():
        name = expected_gguf_name(ckpt_name)
        if name is None:
            print(f"UNMAPPED checkpoint tensor: {ckpt_name}")
            return 1
        expected[name] = list(reversed(torch_shape))

    ok = True
    missing = set(expected) - set(gguf_tensors)
    extra = set(gguf_tensors) - set(expected)
    if missing:
        ok = False
        print(f"MISSING in GGUF ({len(missing)}): {sorted(missing)}")
    if extra:
        ok = False
        print(f"EXTRA in GGUF ({len(extra)}): {sorted(extra)}")

    for name in sorted(set(expected) & set(gguf_tensors)):
        if expected[name] != gguf_tensors[name]:
            ok = False
            print(f"SHAPE MISMATCH {name}: expected ne={expected[name]}, got {gguf_tensors[name]}")

    # headless-draft asserts
    for forbidden in ("token_embd", "token_embd.weight", "output", "output.weight", "d2t", "d2t.weight"):
        if forbidden in gguf_tensors:
            ok = False
            print(f"FORBIDDEN tensor present: {forbidden}")

    def meta(key):
        f = reader.get_field(key)
        return None if f is None else f.contents()

    arch = meta("general.architecture")
    print(f"arch: {arch}, tensors: {len(gguf_tensors)} (expected {len(expected)})")

    if meta("dflash.block_size") != 8:
        ok = False
        print(f"dflash.block_size = {meta('dflash.block_size')} != 8 (STOP GATE risk 5)")
    else:
        print("dflash.block_size = 8 ✓")

    layers = meta("dflash.target_layers")
    if list(layers or []) != [6, 15, 25, 34, 43]:
        ok = False
        print(f"dflash.target_layers = {layers} != [6, 15, 25, 34, 43] (STOP GATE risk 2: off-by-one)")
    else:
        print("dflash.target_layers = [6,15,25,34,43] ✓")

    for k, want in [("dflash.conv_kernel_size", 2), ("dflash.conv_group_size", 16),
                    ("dflash.selector_rank", 256), ("dflash.selector_top_k", 16)]:
        got = meta(k)
        if got != want:
            ok = False
            print(f"{k} = {got} != {want}")
        else:
            print(f"{k} = {got} ✓")

    if not ok:
        print("INVENTORY CHECK FAILED")
        return 1
    print(f"INVENTORY CHECK PASSED ({len(gguf_tensors)} tensors)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
