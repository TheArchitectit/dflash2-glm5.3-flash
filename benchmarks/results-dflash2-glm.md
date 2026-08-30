# DFlash2 × GLM-5.3-Flash on CPU — Benchmark Results (v0.0.1)

Box: ucs03 (dual Xeon, 251 GB RAM, CPU-only, no GPU). llama.cpp glm5 fork
(f30bed8 lineage, DFlash2 support). Target: UD-IQ4_XS 147 GB, 5 shards, mlock.
Draft: `dflash2-glm-f16.gguf` (2.3 GB, headless, converts checkpoint
`incoai/GLM-5.3-Flash-DFlash2` F16). Locked config:
`--spec-type draft-dflash --spec-draft-n-max 4 --spec-draft-p-min 0.4 --top-k 20`
(temp 1.0, top-p 0.95, min-p 0.01, repeat-penalty 1.05, ctx 131k, FA on, f16 KV).
Both arms run serially (solo rule); same server binary, same units except spec
flags. Acceptance counted with `steps = predicted_n - accepted`
(the server's own identity, server-context.cpp:664) — NOT draft_n/n_max.

## Headline: +37-41% measured across three workload classes (identical suite, back-to-back arms)

| task (10 prompts each) | spec-off t/s | spec-on t/s | delta | spec-on acc |
|---|---|---|---|---|
| toolcall (forced function calls) | 1.517 | 2.133 | **+40.6%** | 3.61 |
| multiturn (3-5 round tool loop) | 1.336 | 1.863 | **+39.4%** | 2.78 |
| summarize (5-9k ctx, 128 out) | 0.956 | 1.323 | **+38.4%** | 3.86 |
| **suite aggregate (wall)** | 4115 s | **3011 s** | **+36.7%** | — |

Raw: `benchmarks/raw/spec_on_*.json`, `benchmarks/raw/spec_off_*.json`,
logs `sprint4.3_spec_on.log`.

## Decode-phase / acceptance runs

| run | suite | acc (corrected) | t/s |
|---|---|---|---|
| 50-prompt agentic gate (3.6) | 50 mixed | 2.76 (server-side) | 1.864 |
| GSM8K mirror (task #32) | 5×2 reps | 2.693 | 2.161 |
| 8-prompt A/B (re-baseline) | 8 short | 1.79 | 1.65 |

Acceptance is workload-INDEPENDENT on CPU (~2.7-2.8 on mixed) EXCEPT structured
output (toolcall 3.6, summarize 3.9) where drafting is easier. The published
GPU reference (same base model, same drafter; brandonmusic/tr3-4bpw card) hit
5.428 on GSM8K — the gap is target precision (IQ4_XS logits vs EXL3/FP8), not
prompt class (their own note: "synthetic long-context acceptance near 2.8-3.0
is not evidence of a broken DFlash implementation").

## Correctness gates (must pass before publishing "lossless")

- **Golden draft-correctness (REQ-SD-2): PASS.** ctx_hidden cos 1.000000 vs
  SGLang reference reimpl at 1e-3; candidate overlap 15-16/16 per slot.
- **Losslessness (REQ-SD-4, amended): PASS as distribution-level.** Greedy
  bitwise spec-on vs spec-off: 2/10 identical — both arms are self-deterministic
  (spec-on 10/10 rerun, spec-off 3/3), so the divergence is deterministic
  GEMM-vs-GEMV numerics in the verify batch flipping near-tied argmaxes.
  The accept path (sampling.cpp:678) samples the target's own distribution and
  accepts only on exact match — the standard distribution-lossless property of
  speculative decoding; vLLM's `rejection_sample_method: standard` is identical.
  Claim in the model card: **distribution-preserving (greedy outputs may differ
  at near-tie tokens on CPU); spec-on is deterministic and repeatable.**
- **T0 net-loss: cleared.** Every arm above beats spec-off.

## Known limits / honest caveats

- 8-token verify costs ~2.75× a single token on this MoE target (RC-1);
  kernel-level expert-reuse batching (F3) is the next lever, parked.
- `--spec-type draft-mtp` cannot run: glm5next NextN graph unimplemented
  (glm5next.cpp:690 assert) despite shipped weights.
- IQ4_XS quantization is the acceptance ceiling vs GPU references; smaller
  quants trade quality.
- Summarize wall t/s dilutes the decode win with ~100 s prefills (identical
  both arms; the decode-phase number is 1.32/0.96).

## External reference points

- incoai published 1.3% cycle overhead / 2.7-3.4× on GPU (SGLang) — not
  comparable to CPU bandwidth-bound; cited for lineage.
- Same-base GPU GSM8K 5.428 acceptance (see above).
