# DFlash2 for llama.cpp — OpenSpec (rev 2, post-research)

> **Research-driven revision.** The original spec (rev 1, git history) assumed a full
> from-scratch port. Deep research (6 parallel agent reports in `research/`) revealed
> DFlash2 support was already merged into the llama.cpp lineage both our forks carry
> (PR #27342), including the grouped dynamic conv and candidate selector. The work is
> therefore **conversion + validation**, not porting.

## Purpose

Run incoai's DFlash2 block-diffusion speculative decoding for GLM-5.3-Flash on the
CPU-only ucs03 box. Projected: 1.32 t/s baseline → ~7 t/s at the published 5.78
accepted tokens/step (bandwidth-bound CPU: verification of N tokens costs one weight
read).

## Ground truth sources

- Checkpoint: `incoai/GLM-5.3-Flash-DFlash2` (BF16 safetensors, 81 tensors, 2.2 GB)
- Reference impl: SGLang PR #36708 (installed in
  `/mnt/ollama/models/glm-5.3-flash/sglang-venv/.../sglang/srt/models/dflash.py`)
- llama.cpp impl: `/mnt/ollama/models/llama-cpp-glm5/src/models/dflash.cpp` (988 lines)
- Spec framework: `/mnt/ollama/models/llama-cpp-glm5/common/speculative.cpp` (dflash impl at ~line 962)
- Converter: `/mnt/ollama/models/llama-cpp-glm5/conversion/qwen.py` (dflash_config handling at ~line 674-710)
- Research reports: `research/01..06-*.md` (this repo)

## Verified facts (from research + code inspection)

1. **glm5next target supports arbitrary-layer extraction.**
   `llama_set_embeddings_layer_inp(ctx_tgt, id, true)` works on glm5next at any layer;
   mHC multi-stream hidden states auto-collapse via `build_hc_mean` to a single
   n_embd vector — the exact representation the DFlash2 checkpoint was distilled
   against. No new hooks needed. (research/04)
2. **The dflash spec impl accepts 5 target layers.** The `target_layer_ids_n != 3`
   assert belongs to the *Eagle3* impl only; the DFlash impl asserts `> 0`. (research/03, speculative.cpp:962)
3. **The model class GQA branch matches our checkpoint exactly**: q/k/v/o proj +
   per-head q_norm/k_norm (RMSNorm over head_dim 128, applied BEFORE RoPE — verified
   in SGLang's Triton kernel). (research/01)
4. **Grouped dynamic conv tensors already exist**: `attn_conv_base` [n_embd, kernel, 2],
   `attn_conv_proj`, and the selector trio. Note the GGUF conv base layout is
   `[n_embd, kernel, 2]` vs checkpoint's `[2, 2, 4096]` (side, tap, channel) — the
   converter must transpose accordingly. (research/05, dflash.cpp:235-238)
5. **Layer indexing is off by one**: llama.cpp `target_layers` metadata is 1-indexed;
   the HF `dflash_config.target_layer_ids` is 0-indexed. The existing converter
   applies `+1`. Our [5,14,24,33,42] must become [6,15,25,34,43]. A silent off-by-one
   here produces plausible-but-garbage drafts. (research/05, qwen.py:709)
6. **Draft KV is materialized, not built**: the worker projects target hidden states
   through the draft's KV heads and writes them into the draft KV cache directly
   (the `batch_inject` embd path). The cafe/glm5 framework implements this via
   `llama_decode` with an embd batch. (research/02, research/03)
7. **Unary logits borrow the target's lm_head** (`_project_candidate_logits`); the
   selector's top_k=16 walk (`_score_edges` + `_follow_maps`) then picks the coherent
   7-token path. The llama.cpp impl already structures this. (research/01, research/03)
8. **Reference GGUF exists**: `incoai/Qwen3.8-27B-DFlash2-GGUF` (Q4_K_M) — known-good
   output of the same converter path; use it to diff metadata/tensor naming.
   (research/06)
9. **block_size=8** for our checkpoint (7 draft tokens/step). The impl default is 16;
   model metadata must always win. (research/02)

## Requirements

### REQ-1: Converter run + parity (primary deliverable)

Run the glm5 fork's converter on `incoai/GLM-5.3-Flash-DFlash2`:

- Vocab: from the GLM-5.3-Flash GGUF/target dir (draft has none) — eagle3-style
  `set_vocab` from target_model_dir
- Metadata: `dflash.block_size=8`, `conv_kernel_size=2`, `conv_group_size=16`,
  `selector_rank=256`, `selector_top_k=16`, `target_layers=[6,15,25,34,43]` (+1 applied),
  `mask_token_id=154856`
- Tensor names per the existing mapping (fc, enc_output_norm, blk.N.*, selector_*)
- **Parity gates:**
  a. Tensor inventory + shapes match rev-1 table (81 tensors)
  b. Metadata diff vs `incoai/Qwen3.8-27B-DFlash2-GGUF` shows only model-specific
     differences
  c. Conv base transposed from [2,2,4096] to [4096,2,2]... verify actual expected
     layout [n_embd, kernel, 2] with a golden test
- If the converter rejects `DFlash2DraftModel` arch name: register the alias
  (`@ModelBase.register("DFlash2DraftModel")`) — smallest possible patch

### REQ-2: Load + spec-decode smoke test

- `llama-server` (glm5 fork) + GLM-5.3-Flash IQ4_XS target + `--spec-type draft-dflash
  -md dflash2-glm.gguf`
- Gates: server starts, draft_n > 0 in timings, no asserts
- The `!= 3`/5-layer question is already resolved (fact 2) — no fork patch expected

### REQ-3: Correctness validation

- Golden test: run SGLang reference (CPU, torch) on a canned prompt, dump draft
  hiddens + proposed path; replay in llama.cpp; match within 1e-3 rel
- End-to-end: acceptance length on ~50 agentic prompts; gate ≥ 5.0 (published 5.78)
- Sanity: greedy outputs with spec on == greedy outputs with spec off (lossless claim)

### REQ-4: CPU benchmark + publish

- ucs03 solo run: t/s vs 1.32 baseline, acceptance %, wall-clock on the standard
  3-task agentic suite
- Publish: GGUF on HF (CC BY-NC-ND 4.0 attribution), notes to incoai + llama.cpp

## Risks (updated)

| # | Risk | Mitigation |
|---|---|---|
| 1 | Converter doesn't know `DFlash2DraftModel` / `glm5_next` arch names | Register alias; 1-line patch |
| 2 | Off-by-one layer ids silently corrupt drafting | Fact 5; verify against reference GGUF metadata + golden test |
| 3 | Conv base tensor layout mismatch ([2,2,4096] vs [n_embd,kernel,2]) | Explicit golden test on conv output |
| 4 | mHC collapse (`build_hc_mean`) doesn't match SGLang's extraction for GLM | Golden test REQ-3; if mismatch, hook SGLang's exact reduction |
| 5 | block_size default 16 overrides trained 8 | Assert metadata present in converter output |
| 6 | License: CC BY-NC-ND 4.0 weights | Non-commercial HF upload with attribution; converter is ours (MIT) |
| 7 | mHC collapse mismatch CONFIRMED: llama.cpp `build_hc_mean` = unweighted mean (models.h:1350) vs SGLang's learned gated contraction (mhc.py:1626) — acceptance degraded | Sprint 5.3 golden test quantifies; 5.4 patches the dflash extraction path with the gated contraction; 1e-3 gate before claiming fixed |
| 8 | MoE verify-cost blowup: 8-token verify batch reads ~2.7× a single token's weights (expert-routing spread) | n_max reduction + p_min gating (Sprint 5.2); config-only path to ~+59% |

## Phases

- **Phase A (DONE 2026-08-29)**: REQ-1 — convert, diff vs reference GGUF. All
  parity gates passed; arch alias already registered (qwen.py:642).
- **Phase B (DONE 2026-08-29)**: REQ-2 — smoke test passed (draft_n>0, no
  asserts). Measured: 1.5 t/s (+14%), acceptance 3.36/7 — below the 5.78
  projection. Gap analysis in research/07-gap-analysis.md.
- **Phase C**: REQ-3 — golden + acceptance validation. Now doubles as the
  verification gate for the mHC-collapse fix (Sprint 5.3–5.4): llama.cpp's
  `build_hc_mean` is an unweighted mean (models.h:1350) while SGLang trained
  the draft on gated contractions (mhc.py:1626) — confirmed divergence to fix.
- **Phase D**: REQ-4 — benchmark, publish.
- **Phase E (Sprint 5)**: gap closure to +60% (≥2.1 t/s): synth-rate
  calibration, config sweep (n_max, p_min, top-k, threads), mHC extraction fix,
  lossless re-validation, final benchmark.
