# Community notes — DRAFTS ONLY (posting left to the user)

## 1. z-lab/dflash (GitHub discussion/issue)

> **GLM-5.3-Flash DFlash2 draft runs on CPU-only llama.cpp — measured +41%**
>
> We converted `incoai/GLM-5.3-Flash-DFlash2` (81 tensors, headless) to GGUF
> via the merged converter path and ran it against the IQ4_XS target on
> llama.cpp (`--spec-type draft-dflash`), CPU only (2× Xeon, 251 GB).
>
> Results (all identical-prompt serial A/B, raw dumps public):
> - 50-prompt agentic: 1.864 t/s (+41% over the 1.32 t/s no-spec baseline)
> - GSM8K mirror of the tr3-4bpw card's table: acc 2.69 vs their GPU 5.43 —
>   same prompts, same drafter → the gap is target precision, not workload
> - 3-task suite: +40.6% toolcall / +39.4% multiturn / +38.4% summarize
> - Golden correctness vs a faithful SGLang reimpl: ctx_hidden cos 1.0 at
>   1e-3, candidate overlap 15-16/16
>
> Config findings worth carrying upstream: on MoE targets `n_max 4 + p_min
> 0.4` beats full 7-token blocks end-to-end (+32% t/s), and top-k 40→20 was
> worth +51%. Follow-up perf work (see note 3) measured the verify cost as
> essentially flat in block width — weight traffic amortizes — so short
> blocks win mainly via acceptance gating, not avoided expert re-reads.
> Acceptance is ~2.7-3.6 on this target/quant across workload classes.
> Full data + methodology: https://github.com/TheArchitectit/dflash2-llamacpp

## 2. ggml-org/llama.cpp (discussion comment)

> **DFlash2 spec-decode confirmed working for a glm5next target on CPU**
> (PR #27342 path). Notes from a 147 GB IQ4_XS run:
> - `target_layers` GGUF metadata is 1-indexed (converter +1); off-by-one
>   produces plausible garbage, worth an explicit converter assert.
> - Headless draft (`ctx_other` embedding/lm_head borrow) works on CPU with
>   `llama_set_embeddings_nextn(ctx, true, /*masked*/ false)` — masked=true
>   zeroes the lattice rows and trips the encoder-graph pooling assert.
> - mHC capture semantics: the DFLASH capture is the **unweighted stream
>   mean** (`hc_contract`), NOT the gated `_mhc_pre` contraction used inside
>   layers — `build_hc_mean` is correct for the capture; a "fix" to gated
>   contraction measured acceptance 3.16→no change and was reverted.
> - `--spec-type draft-mtp` cannot start on glm5next: the fork hard-asserts
>   the NextN decoder graph unimplemented (models/glm5next.cpp:690) even
>   though `nextn_predict_layers=[1]` weights ship in the target GGUF —
>   implementing it would let these models use their built-in drafter.
> - Distribution-losslessness holds; bitwise greedy spec-on==spec-off does
>   NOT on CPU (verify GEMM ≠ decode GEMV flips near-ties deterministically).
>
> Bench data + golden harness: https://github.com/TheArchitectit/dflash2-llamacpp

## 3. ggml-org/llama.cpp (issue: missing AVX-512/VNNI IQ3_S path — CPU MoE bottleneck)

> **IQ3_S `vec_dot` has no AVX-512 path — it is the hottest function in
> mixed-IQ MoE decode on CPU**
>
> `perf record` on a GLM-5.3-Flash mixed-IQ target (routed experts =
> 82 IQ3_S + 41 IQ4_XS + a few Q*_K tensors) under 8-token spec-verify:
>
> | symbol | self % |
> |---|---|
> | `ggml_vec_dot_iq3_s_q8_K` | **32.89%** |
> | `ggml_vec_dot_iq4_xs_q8_K` | 5.88% |
> | `gated_delta_net` (KDA) | 1.04% |
>
> The x86 tree has no `_mm512` path for either IQ dot (AVX2 `_mm256_set_epi32`
> grid construction at `arch/x86/quants.c:3384-3478`), while RISC-V already
> ships vl128/256/512 specializations. On AVX-512 VNNI hardware (Ice Lake+,
> Zen4+) a `_mm512` rewrite of the IQ3_S dot — 2× width, real VNNI, and
> possibly a cheaper grid lookup — should plausibly get 1.3-2× on the single
> hottest function of CPU MoE inference. Our measurement box is Haswell
> (AVX2 only, E5-2660 v3) so we cannot quantify; microbench gate we'd use:
> ≥1.3× on IQ3_S×Q8_K at n=4096 widths before touching `mul_mat_id`.
>
> Related measurements (all raw + methodology public): verify cost is flat
> in batch width — `cost(n) = 0.759 + 0.158·(n−1) s` on a 147 GB IQ4_XS/
> IQ3_S target — and an 8-token verify touches 38.3 distinct experts (not
> the ~58 an uncorrelated model predicts), so block-diffusion drafts should
> be evaluated against measured, not modeled, routing. The probe counters
> are 75 lines, env-gated (`LLAMA_MOE_PROBE=1`), on a branch if useful:
> https://github.com/TheArchitectit/dflash2-llamacpp (research/09)
