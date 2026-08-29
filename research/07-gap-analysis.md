# Gap Analysis: +60% Throughput on GLM-5.3-Flash + DFlash2 (CPU)

## Baseline measurements (Sprint 2, 2026-08-29)

| Metric | Value |
|---|---|
| Target decode (baseline, no spec) | 1.32 t/s (758 ms/token) |
| Spec-decode effective | ~1.5 t/s (+14%) |
| Mean accepted length | 3.36 (published for this class: 4.39–5.46 on Qwen3.8-27B; 5.78 headline) |
| Per-position acceptance | (0.72, 0.51, 0.38, 0.26, 0.24, 0.16, 0.10) — heavy decay |
| Verify step cost | ~2.2 s (should be ≈1× weight read ≈ 0.87 s incl. draft) |
| Draft step cost | ~111 ms/block (not the bottleneck) |
| Published DFlash2 throughput | 2.7–3.4× on GPU (SGLang); 1.3% cycle overhead |

## Root causes (ranked)

### RC-1: Verify-cost blowup from MoE routing divergence (biggest lever)
The 8-token verify batch routes to more distinct experts than a 1-token decode.
With 288 experts, 8 active/token: uncorrelated routing would touch ~58 experts
(E≈288·(1−(1−8/288)^8)≈58) ⇒ ~6.5× the FFN weight read. Our measured 2.2 s
(≈2.7× single-token cost) implies strong routing correlation between drafted
tokens, but still far above 1×. Classic spec-decode "verify is free" only holds
for dense models. **This is architectural**, not a bug: the glm5 fork's dflash
impl is the NEWER of the two forks (verified by diff: has is_dflash2, is_mrope,
p_min gating, metadata-driven causal_attn; cafe's copy lacks them all). Nothing
to backport.

Fixes:
- **F1 (config, hours)**: lower `--spec-draft-n-max` from 7 → 4–5. Verify
  batch shrinks 8→5 tokens; MoE spread drops roughly (E(5)≈37 vs E(8)≈58);
  expected cycle: ~1.5 s at ~3.2 accepted ⇒ ~2.1 t/s (**+40%**). We keep the
  tail-accept loss (~0.3 tokens) but those positions only accept 10–16% anyway.
- **F2 (config, minutes)**: `--spec-draft-p-min 0.3–0.5` — the glm5 fork gates
  the selector walk on softmax confidence (speculative.cpp:1281-1299), truncating
  low-confidence tails dynamically. Better than static n_max on mixed workloads.
- **F3 (kernel, weeks)**: expert-reuse-aware verification — batch the verify
  pass so tokens sharing experts amortize reads (requires ggml MoE kernel work,
  or CPU_REPACK interleave). High effort, uncertain payoff. Park.

### RC-2: Acceptance below published (3.36 vs 4.4–5.5)
Candidate causes:
- **Sampling**: published numbers use temp 1.0 + top-p 0.95 + top-k 20 +
  presence-penalty 1.5 with lossless rejection sampling. We run temp 1.0, top-p
  0.95, top-k **40**, min-p 0.01, repeat-penalty 1.05. Wider top-k + min-p
  widens the target's sampled distribution → fewer draft matches.
  **F4 (config, minutes)**: match published sampling (top-k 20; consider
  presence-penalty semantics vs repeat-penalty). Expected +0.3–0.8 accepted.
- **mHC collapse mismatch (risk 4, unverified)**: llama.cpp `build_hc_mean`
  vs SGLang `_mhc_pre` contraction (Sinkhorn-normalized mixing, per-stream
  weights). If they differ, the draft sees a different hidden distribution than
  it was distilled on. **F5 (Sprint 3 golden test, days)**: run the REQ-3
  golden test — dump SGLang draft hiddens on a canned prompt, replay in
  llama.cpp, compare at 1e-3. This is already specced; do it next.
- **Draft quality (F16 vs BF16)**: our GGUF is F16; checkpoint is BF16. F16
  has more mantissa but different rounding — negligible. **F6 (minutes)**:
  re-convert with `--outtype bf16` if golden test shows small drift.

### RC-3: Decode-phase threading inefficiency
Prompt eval ran at 4.63 t/s (216 ms/token at batch 25) — batched decode is NOT
amortizing; consistent with KDA (linear attention) sequential state updates
inside each chunk plus MoE spread. Also 40 logical threads on 20 physical
cores: HT adds sync overhead on memory-bound GEMV.
- **F7 (config, minutes each)**: test `--threads 20`, `--poll 100`,
  `--spec-draft-threads 16`. Expected 0–10%.

### RC-4: Draft overhead is small but nonzero (111 ms/block)
Draft is 2.3 GB F16. Q8_0 halves the read (~19 ms saved/block) but risks
selector precision. **F8 (optional)**: convert draft to Q8_0, A/B the
acceptance. ≤1% expected; do last.

## Combined estimate (config-only: F1/F2 + F4 + F7)

| Scenario | Cycle | Accepted | t/s | vs baseline |
|---|---|---|---|---|
| Now (n_max 7) | 2.24 s | 3.36+1 | ~1.5 | +14% |
| n_max 4 + p_min 0.4 + top-k 20 | ~1.5 s | ~3.3 | ~2.1 | **+59%** |
| + threads tuned | ~1.4 s | ~3.3 | ~2.3 | +74% (if F7 pays) |

**The +60% target is reachable with config changes alone.** The golden test
(F5) is the insurance policy: if mHC collapse is wrong, fixing it could push
acceptance toward the published 4.4–5.5 and t/s toward 2.8–3.2 (2.1–2.4×).

## Verification plan (in order)

1. **V1**: `--spec-synth-rates` calibration run (benchmarking mode): measures
   pure cycle cost at each n_max without acceptance noise — isolates RC-1 from
   RC-2 in one solo run. (glm5 fork supports this; cafe does not.)
2. **V2**: golden test (Sprint 3.1–3.5 as specced).
3. **V3**: acceptance sweep on 50 agentic prompts with tuned config.
4. **V4**: greedy lossless check (spec on == spec off) — mandatory before
   publishing any "faster" claim.

## Sources
- Published tables & methodology: research/06-ecosystem.md (DFlash2 paper/blog,
  SGLang PR #36708, inco.ai blog)
- Fork diff: `diff llama-cpp-glm5/common/speculative.cpp llama-cpp-cafe/common/speculative.cpp`
  (295 lines; glm5 fork is newer — has p_min gating at 1281-1299, is_dflash2,
  is_mrope handling, metadata-driven causal_attn)
- Measured data: journalctl ucs03, Sprint 2 smoke run (2026-08-29)
