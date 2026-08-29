#!/usr/bin/env python3
"""Sprint 1.7 / 1.8 — metadata parity diff: our GGUF vs reference DFlash2 GGUF.

Usage:
    diff_gguf_meta.py <ours.gguf> <ref.gguf>

Diffs all KV metadata. Keys whose values legitimately differ between the GLM
draft and the Qwen reference (model-specific) are ALLOWED; anything else is a
conversion bug. Also asserts dflash.target_layers == [6,15,25,34,43] (task 1.8
off-by-one STOP GATE).

Exit 0 = only allowlisted diffs. Exit 1 = unexpected difference (STOP GATE).
"""
import sys

# keys that are expected to differ (model-specific), with a reason
ALLOWED = {
    "general.name":                     "model name",
    "general.basename":                 "model name",
    "general.size_label":               "model size",
    "general.license":                 "cc-by-nc-nd-4.0 vs apache-2.0",
    "general.base_model":               "different base model",
    "dflash.context_length":            "1048576 vs 262144",
    "dflash.embedding_length":          "4096 vs 5120",
    "dflash.feed_forward_length":       "12288 vs 17408",
    "dflash.rope.freq_base":            "10000 vs 1e7",
    "dflash.attention.layer_norm_rms_epsilon": "1e-5 (GLM) vs 1e-6 (Qwen)",
    "general.file_type":                "F16 vs Q8_0",
    "tokenizer.ggml.tokens":            "different vocab",
    "tokenizer.ggml.scores":            "different vocab",
    "tokenizer.ggml.token_type":        "different vocab",
    "tokenizer.ggml.merges":            "different vocab",
    "tokenizer.ggml.model":             "different tokenizer class",
    "tokenizer.ggml.pre":              "different tokenizer pre",
    "tokenizer.ggml.bos_token_id":     "different vocab",
    "tokenizer.ggml.eos_token_id":      "different vocab",
    "tokenizer.ggml.mask_token_id":    "154856 vs 248070",
    "tokenizer.ggml.padding_token_id": "different vocab",
    "tokenizer.ggml.eom_token_id":     "GLM vocab has it, Qwen ref doesn't",
    "tokenizer.ggml.eot_token_id":      "GLM vocab has it, Qwen ref doesn't",
    "tokenizer.ggml.unknown_token_id": "GLM vocab has it, Qwen ref doesn't",
    "tokenizer.ggml.add_bos_token":    "Qwen ref has it, GLM doesn't",
    "tokenizer.chat_template":          "different chat template",
    "general.author":                   "Inco AI (ref only)",
    "general.organization":             "z-lab (ref only)",
    "general.source.url":               "ref only",
    "general.base_model.0.name":        "different base",
    "general.base_model.0.organization": "different base",
    "general.base_model.0.repo_url":    "different base",
    "general.tags":                     "model-specific tags",
    "general.quantized_by":             "we are the quantizer",
    "general.finetune":                 "model-specific",
    "general.filename":                 "file naming",
    "general.quantization_version":     "Q8_0 vs F16",
    "GGUF.kv_count":                     "structural (vocab keys differ)",
}

# keys that MUST match exactly
MUST_MATCH = {
    "general.architecture":             "dflash",
    "dflash.block_count":               5,
    "dflash.block_size":                8,
    "dflash.conv_kernel_size":          2,
    "dflash.conv_group_size":           16,
    "dflash.selector_rank":             256,
    "dflash.selector_top_k":            16,
    "dflash.attention.head_count":      32,
    "dflash.attention.head_count_kv":   8,
    "dflash.attention.key_length":      128,
    "dflash.attention.value_length":    128,
    "dflash.attention.causal":          False,
    "dflash.attention.sliding_window":  2048,
    "dflash.sliding_window_pattern":    [1, 1, 1, 1, 1],
    "dflash.target_layers":             [6, 15, 25, 34, 43],  # task 1.8 STOP GATE
}


def load(path):
    sys.path.insert(0, "/mnt/ollama/models/llama-cpp-glm5/gguf-py")
    from gguf import GGUFReader
    r = GGUFReader(path)
    out = {}
    for f in r.fields.values():
        if f.name == "general.architecture" or True:
            try:
                out[f.name] = f.contents()
            except Exception as e:
                out[f.name] = f"<unreadable: {e}>"
    return out


def norm(v):
    if isinstance(v, list):
        return tuple(v)
    return v


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    ours, ref = load(sys.argv[1]), load(sys.argv[2])
    ok = True

    only_ours = set(ours) - set(ref)
    only_ref = set(ref) - set(ours)
    for k in sorted(only_ours):
        if k not in ALLOWED and not k.startswith("general.quantized"):
            # tensor-count/size keys are structural, not model-specific
            if k in ("general.file_type",):
                continue
            print(f"KEY ONLY IN OURS: {k} = {str(ours[k])[:60]}")
            ok = False
    for k in sorted(only_ref):
        if k not in ALLOWED and k not in ("general.file_type", "general.file_type_int"):
            print(f"KEY ONLY IN REF:  {k} = {str(ref[k])[:60]}")
            ok = False

    for k in sorted(set(ours) & set(ref)):
        if k in MUST_MATCH:
            want = MUST_MATCH[k]
            got = norm(ours[k])
            if list(got) != list(want) if isinstance(want, list) else got != want:
                ok = False
                print(f"MUST-MATCH FAIL {k}: ours={got}, want={want}")
        elif k in ALLOWED:
            pass  # legit difference
        else:
            if norm(ours[k]) != norm(ref[k]):
                print(f"UNEXPECTED DIFF {k}: ours={str(ours[k])[:50]} ref={str(ref[k])[:50]}")
                ok = False

    # standalone task 1.8 assert
    tl = ours.get("dflash.target_layers")
    if list(tl or []) != [6, 15, 25, 34, 43]:
        ok = False
        print(f"TASK 1.8 FAIL: dflash.target_layers = {tl} != [6,15,25,34,43] (off-by-one!)")

    if not ok:
        print("METADATA DIFF FAILED")
        return 1
    print("METADATA DIFF PASSED (only allowlisted model-specific differences)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
