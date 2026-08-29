# Design: add-dflash2-support

## Context

DFlash2 support was merged into the llama.cpp lineage both forks carry (PR #27342, commit
b10f9ca58, Aug 27 2026). The components that already exist:

- **Model class**: `/mnt/ollama/models/llama-cpp-glm5/src/models/dflash.cpp` (988 lines).
  `LLM_ARCH_DFLASH` with the GQA branch matching our checkpoint (q/k/v/o proj + per-head
  q_norm/k_norm, RMSNorm over head_dim 128 before RoPE). Grouped dynamic conv tensors
  already declared: `dflash_attn_conv_base` `{n_embd, kernel, 2}`, `dflash_attn_conv_proj`,
  `dflash_ffn_conv_base`, `dflash_ffn_conv_proj` (dflash.cpp:235-238); consumed by
  `build_dflash2_conv` (dflash.cpp:404-477) and `build_dflash2_selector` (dflash.cpp:479+).
- **Spec framework**: `/mnt/ollama/models/llama-cpp-glm5/common/speculative.cpp`,
  `common_speculative_impl_draft_dflash` at ~line 962. DFlash impl asserts
  `target_layer_ids_n > 0` (speculative.cpp:962) — accepts our 5 layers; the `!= 3` assert
  belongs to the Eagle3 impl only. `is_dflash2` is keyed on `selector_top_k > 0`
  (speculative.cpp:984). Block size read from the `dflash.block_size` metadata key with a
  hard-coded default of 16 (speculative.cpp:969-975). Draft KV is materialized, not built:
  target hidden states are projected and written into the draft KV cache via the
  `batch_inject` embd path (`llama_decode` with an embd batch).
- **Converter**: `/mnt/ollama/models/llama-cpp-glm5/conversion/qwen.py` — `DFlashModel`
  class at line 642, registered for both `"DFlashDraftModel"` and `"DFlash2DraftModel"`
  (`@ModelBase.register` at qwen.py:642, `model_arch = gguf.MODEL_ARCH.DFLASH`). Handles
  `dflash_config` at qwen.py:674-727: `block_size`, `conv_kernel_size`, `conv_group_size`,
  `selector_rank`, `selector_top_k`, `mask_token_id`, and target layers with `+1`
  applied at qwen.py:707-710 (`extract_layer_ids = [i + 1 for i in target_layer_ids]`).
  Vocab delegation: `set_vocab` (qwen.py:647-673) requires `--target-model-dir` and
  reuses the target model's own vocab handler via `get_model_class(target_arch)`.
- **Target hooks**: glm5next supports extraction at arbitrary layers via
  `llama_set_embeddings_layer_inp(ctx_tgt, id, true)`; mHC multi-stream states
  auto-collapse via `build_hc_mean` (glm5next.cpp:613) to a single n_embd vector — the
  representation the draft was distilled against (research/04).
- **Reference GGUF**: `incoai/Qwen3.8-27B-DFlash2-GGUF` (Q4_K_M) — known-good output of
  the same converter path; use for metadata/tensor-naming diff.
- **Checkpoint**: `incoai/GLM-5.3-Flash-DFlash2` (BF16 safetensors, 81 tensors, 2.2 GB),
  `dflash_config`: `block_size=8`, `target_layer_ids=[5,14,24,33,42]` (0-indexed HF),
  `mask_token_id=154856`, `conv_kernel_size=2`, `conv_group_size=16`, `selector_rank=256`,
  `selector_top_k=16`.
- **Target**: GLM-5.3-Flash IQ4_XS (147 GB) on CPU-only ucs03, 1.32 t/s baseline.

## Goals / Non-Goals

**Goals**

- Convert `incoai/GLM-5.3-Flash-DFlash2` to a GGUF the glm5 fork loads as a `draft-dflash`
  drafter, with metadata/tensor parity proven against the reference GGUF.
- Validate correctness: golden test vs SGLang reference, acceptance gate, greedy losslessness.
- Measure effective t/s on ucs03 against the 1.32 t/s baseline; publish the GGUF.

**Non-Goals**

- No porting: no new C++ model class, no new spec impl, no new converter architecture.
- No upstream llama.cpp contribution as a deliverable (the support is already merged).
- No SGLang changes; it is used read-only as the reference implementation.
- No training, no draft quality improvement, no quantization scheme search beyond using the
  checkpoint as-is (BF16 → GGUF F16/quant decided by the existing converter path).
- No changes to the production `:8086` service; benchmarks run solo.

## Decisions

### D1: Use the glm5 fork's existing converter, not a from-scratch `convert_dflash2_to_gguf.py`

The fork's `DFlashModel` (conversion/qwen.py:642-775) already implements everything rev-1
planned to write: arch registration for `DFlash2DraftModel`, `dflash_config` metadata
extraction (qwen.py:674-727), the +1 target-layer conversion (qwen.py:707-710), vocab
delegation from `--target-model-dir` (qwen.py:647-673), and the checkpoint→GGUF tensor
mapping (`gguf-py/gguf/tensor_mapping.py:1358-1377` maps `attention_conv.base_kernel` /
`kernel_projection` / selector codebooks). Writing a parallel converter would duplicate this
and risk divergence. The original rev-1 plan (`convert_dflash2_to_gguf.py` in this repo) is
dropped.

### D2: Vocab sourced from the GLM-5.3-Flash target dir — eagle3-style `set_vocab`

The draft checkpoint has no tokenizer or embedding table. The converter's `set_vocab`
(qwen.py:647-673) already raises unless `--target-model-dir` is given, then delegates to the
target arch's own vocab handler by reading the target's `config.json` and looking up
`get_model_class(target_arch)` — which resolves to `Glm5NextModel`
(conversion/glm5next.py:16-18). No new vocab code. The draft's unary logits borrow the
target's lm_head at runtime (`_project_candidate_logits`), consistent with a headless draft.

### D3: Layer id +1 conversion: `[5,14,24,33,42]` → `[6,15,25,34,43]`

llama.cpp `target_layers` metadata is 1-indexed; the HF `dflash_config.target_layer_ids` is
0-indexed. The existing converter applies `+1` (qwen.py:707-710). Our checkpoint's
`[5,14,24,33,42]` must land as `[6,15,25,34,43]` in the GGUF. A silent off-by-one here
produces plausible-but-garbage drafts — this is the single most dangerous conversion step,
so it is verified twice: against the reference GGUF metadata diff and by the golden test
(REQ-CONV-4 scenario).

### D4: Conv base tensor transpose: `[2,2,4096]` → `[4096,2,2]`

The checkpoint stores `attention_conv.base_kernel` as `[2, 2, 4096]` (side, tap, channel);
the GGUF/consumer layout is `{n_embd, kernel, 2}` = `[4096, 2, 2]` (channel, tap, side),
created at dflash.cpp:235 and consumed by `build_dflash2_conv`, which views the base as
`[group_size, n_groups, kernel]` per side (dflash.cpp:439-442) and indexes the side via
`side * base->nb[2]`. The converter must transpose `(side, tap, channel) → (channel, tap,
side)`. The fork's `DFlashModel.modify_tensors` (qwen.py:757-775) has no explicit conv-base
transpose, so this is verified by the golden test on conv output (REQ-CONV-4) rather than
assumed; if the generic mapping already lands the right layout, the golden test proves it
either way.

### D5: `block_size=8` from model metadata must always beat impl default 16

The spec impl hard-codes `block_size = 16` as its fallback (speculative.cpp:969) and only
overrides it when the `dflash.block_size` metadata key parses (speculative.cpp:970-975). Our
checkpoint is trained with block_size=8 (7 draft tokens/step); running at 16 would use
untrained block positions and degrade acceptance. The converter must emit
`dflash.block_size=8` (qwen.py:681-683 defaults to 16 only if the config lacks it — ours has
it), and the smoke test must assert the loaded value.

### D6: Validation strategy — golden test vs SGLang reference, acceptance gate, greedy lossless

Three independent gates, cheapest first:

1. **Golden test (correctness)**: run the SGLang reference
   (`/mnt/ollama/models/glm-5.3-flash/sglang-venv/.../sglang/srt/models/dflash.py`) on CPU
   with torch against a canned prompt; dump draft hiddens + the proposed 7-token path;
   replay the same inputs in llama.cpp; match within 1e-3 relative tolerance. This is the
   only check that catches the silent killers (D3 off-by-one, D4 conv layout, mHC collapse
   mismatch) before they can masquerade as "low acceptance". If mHC `build_hc_mean`
   extraction diverges from SGLang's, the fallback is hooking SGLang's exact reduction.
2. **Acceptance gate (quality)**: mean accepted tokens/step over ~50 agentic prompts must be
   ≥ 5.0 (published 5.78). Below gate → halt before benchmark/publication.
3. **Greedy lossless check**: greedy outputs with spec on == greedy outputs with spec off.
   Any divergence is a correctness bug, not a tolerance question.

### D7: If the `DFlash2DraftModel` arch name is rejected, register an alias via `@ModelBase.register`

The smallest possible patch: one name added to the existing decorator. Code inspection shows
qwen.py:642 already registers `"DFlash2DraftModel"`, so this is pure contingency — planned
for, likely unneeded. If the checkpoint's `config.json` uses an unexpected variant name,
the fix is adding that variant to the same `@ModelBase.register(...)` line. No new converter
logic, no C++ changes.

### D8: Validation harness lives in this repo (`scripts/`); llama.cpp fork untouched where possible

Golden replay, acceptance runner, and benchmark driver go in
`/mnt/ollama/git/dflash2-llamacpp/scripts/`. The fork is exercised only through its public
interfaces (`convert_hf_to_gguf.py`, `llama-server` flags, GGUF metadata). This keeps the
fork diffable against upstream (clean for future pulls) and keeps all project-specific
scripts reviewable in one place. The only fork edit allowed under this design is the D7
1-line alias, which we expect not to need.

## Risks / Trade-offs

| # | Risk | Mitigation |
|---|---|---|
| 1 | Converter doesn't know `DFlash2DraftModel` / `glm5_next` arch names | qwen.py:642 already registers `DFlash2DraftModel`; contingency is the 1-line `@ModelBase.register` alias (D7) |
| 2 | Off-by-one layer ids silently corrupt drafting | D3: +1 applied by existing converter (qwen.py:707-710); verified against reference GGUF metadata + golden test |
| 3 | Conv base tensor layout mismatch ([2,2,4096] vs [n_embd,kernel,2]) | D4: explicit golden test on conv output; transpose (side,tap,channel)→(channel,tap,side) verified, not assumed |
| 4 | mHC collapse (`build_hc_mean`) doesn't match SGLang's extraction for GLM | Golden test (D6) catches it; fallback: hook SGLang's exact reduction. Note SGLang needed follow-up PR #36755 for exactly this on mHC models |
| 5 | block_size default 16 overrides trained 8 | D5: assert `dflash.block_size=8` present in converter output and loaded value in smoke test |
| 6 | License: CC BY-NC-ND 4.0 weights | Non-commercial HF upload with attribution; converter code is ours (MIT) |

## Migration Plan

Phases from spec.md; each phase gates the next.

- **Phase A (now) — conversion + parity (REQ-1)**: run the converter (D1) with
  `--target-model-dir` pointing at the GLM-5.3-Flash dir (D2); verify 81-tensor inventory,
  metadata diff vs `incoai/Qwen3.8-27B-DFlash2-GGUF`, layer ids [6,15,25,34,43] (D3), conv
  base layout (D4), `dflash.block_size=8` (D5). Register alias only if conversion fails (D7).
- **Phase B — smoke test (REQ-2)**: `llama-server` with IQ4_XS target +
  `--spec-type draft-dflash -md dflash2-glm.gguf`; gates: server starts, `draft_n > 0` in
  timings, no asserts.
- **Phase C — correctness validation (REQ-3)**: golden test vs SGLang reference (1e-3 rel
  tol), acceptance ≥ 5.0 over ~50 agentic prompts, greedy lossless check. Any failure halts
  spec-decode work until resolved.
- **Phase D — benchmark + publish (REQ-4)**: ucs03 solo run (production `:8086` idle): t/s vs
  1.32 baseline, acceptance %, wall-clock on the standard 3-task agentic suite. Publish GGUF
  to HF with CC BY-NC-ND 4.0 attribution; notes to incoai + llama.cpp.

Rollback is trivial at every phase: the fork is unmodified (D8), the GGUF is a new artifact,
and no existing service is touched.
