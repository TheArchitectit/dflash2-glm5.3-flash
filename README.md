# dflash2-llamacpp

Port of [incoai's DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2) block-diffusion draft model to [llama.cpp](https://github.com/ggml-org/llama.cpp), enabling speculative decoding for GLM-5.3-Flash on CPU-only hosts.

## Why

DFlash2 is currently SGLang-only (GPU). But block-diffusion speculative decoding is uniquely suited to **bandwidth-bound CPU inference**: verifying `n` draft tokens costs one forward pass through the target model — the same weight read as generating 1 token. With DFlash2's reported 5.78 accepted tokens per verification step, a GLM-5.3-Flash CPU deployment goes from 1.32 t/s to a projected ~7 t/s.

No GGUF conversion of DFlash2 exists. This repo provides:

1. **`convert_dflash2_to_gguf.py`** — safetensors → GGUF converter
2. **Patches against llama.cpp** — new `dflash2` model class + spec-decode wiring
3. **Benchmarks** — acceptance rates and effective t/s vs baseline decode

## Status

- [ ] Phase 0: repo, spec, reference-tensor extraction (this document)
- [ ] Phase 1: GGUF converter + loader parity test
- [ ] Phase 2: single-forward inference test (prefill-only, no spec decode)
- [ ] Phase 3: speculative decode integration
- [ ] Phase 4: CPU benchmarks vs baseline

## Source material

| Item | Location |
|---|---|
| DFlash2 weights (BF16 safetensors) | `incoai/GLM-5.3-Flash-DFlash2` on HF |
| DFlash2 architecture reference | SGLang PR [#36708](https://github.com/sgl-project/sglang/pull/36708) |
| GLM-5.3-Flash GGUF (target model) | `unsloth/GLM-5.3-Flash-GGUF` (UD-IQ4_XS) |
| llama.cpp fork with `glm5next` arch | `unslothai/llama.cpp` branch `glm5next/upstream` |
| llama.cpp fork with dflash spec framework | `quimmedes/cafe-llama.cpp` |
| DFlash blog | https://inco.ai/blog/dflash2/ |

## License notes

DFlash2 weights: CC BY-NC-ND 4.0 (research/eval only — no commercial use without inco.ai license). Code in this repo: MIT.
