---
license: cc-by-nc-nd-4.0
base_model:
  - incoai/GLM-5.3-Flash-DFlash2
  - zai-org/GLM-5.3-Flash
tags:
  - gguf
  - llama.cpp
  - speculative-decoding
  - dflash2
---

# GLM-5.3-Flash DFlash2 draft — GGUF (CPU-validated)

Draft model weights for DFlash2 block-diffusion speculative decoding of
**GLM-5.3-Flash**, converted from `incoai/GLM-5.3-Flash-DFlash2` (BF16, 81
tensors) to GGUF for llama.cpp `--spec-type draft-dflash`. The draft is
**headless** — it borrows the target's token embeddings and lm_head at
runtime (no `token_embd`/`output` tensors).

This repo publishes the **draft only**. The 147 GB target (e.g.
`GLM-5.3-Flash-UD-IQ4_XS`) is served from its own source.

## Files

| file | size | note |
|---|---|---|
| `dflash2-glm-f16.gguf` | 2.35 GB | recommended on CPU |
| `dflash2-glm-q8_0.gguf` | 1.25 GB | half-size; acceptance verified equal-or-better vs F16 on CPU |
| `dflash2-glm-bf16.gguf` | 2.35 GB | exact-parity conversion |

## Serving recipe (validated on CPU, 2× Xeon, 251 GB, llama.cpp glm5 fork)

```
llama-server \
  -m GLM-5.3-Flash-UD-IQ4_XS-00001-of-00005.gguf \
  -md dflash2-glm-f16.gguf \
  --spec-type draft-dflash \
  --spec-draft-n-max 4 --spec-draft-p-min 0.4 \
  --threads -1 --numa distribute --load-mode mlock \
  --ctx-size 131072 --flash-attn on --jinja \
  --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.01 --repeat-penalty 1.05
```

`n_max=4 + p_min=0.4 + top-k 20` beat full 7-token blocks on CPU: MoE verify
cost scales with batch width, so trimming the tail wins end-to-end
(+32-41% t/s measured).

## Measured (CPU, IQ4_XS target)

| suite | t/s | vs baseline 1.32 |
|---|---|---|
| 50-prompt agentic | 1.864 | **+41%** |
| GSM8K (5×2) | 2.161 | **+64%** |
| toolcall / multiturn / summarize arms | +40.6% / +39.4% / +38.4% | identical-prompt A/B |

Acceptance ~2.7-3.6 tok/step on this target (vs 5.428 published on GPU — the
gap tracks target precision, not the drafter; see the golden correctness
chain). Losslessness is distribution-level (the accept path samples the
target's own distribution); each arm is fully self-deterministic on CPU.

Correctness + benchmark provenance: https://github.com/TheArchitectit/dflash2-llamacpp
(`benchmarks/results-dflash2-glm.md`, `tests/golden/`, raw JSON in
`benchmarks/raw/`).

## License

Weights derived from `incoai/GLM-5.3-Flash-DFlash2` — **CC-BY-NC-ND-4.0**
(non-commercial, no derivatives). Target model per zai-org terms.
Conversion/validation code in the GitHub repo is MIT.
