# Improvement Tracking — DFlash2 CPU throughput over time

Every measured config lands here exactly once, newest row on top. Tiers are
derived from the ladder in `benchmarks/acceptance-gate.md` — never hand-edited.

## 8-prompt A/B suite (scripts/bench_acceptance.py defaults, /tmp/ab_bench.py)

| Date | Config | acc_len | t/s | Tier(t/s) | Tier(acc) | Δ vs prev |
|---|---|---|---|---|---|---|
| 2026-08-29 | n_max 4 + p_min 0.4 + top-k 20 | 3.91 | 1.44 | T2 WIN | T3 TARGET | **+32% t/s, +64% acc** |
| 2026-08-29 | n_max 7 + top-k 20 (baseline) | 2.38 | 1.09 | — | T2 WIN | — |

## Controlled 4-prompt sampling sweeps

| Date | Config | acc_len | t/s | Note |
|---|---|---|---|---|
| 2026-08-29 | top-k 20 | 2.65 | 1.25 | F4 confirmed vs top-k 40 |
| 2026-08-29 | top-k 40 | 1.94 | 0.83 | too wide |
| 2026-08-29 | presence-penalty 1.5 | 1.74 | — | HURTS (vs 2.49 without) |
| 2026-08-29 | top-p 0.85 | — | — | marginal over 0.95 |

## Sprint 5.1 synth-rate calibration (forced acceptance, no noise)

| Date | Config | cycle | t/s | note |
|---|---|---|---|---|
| 2026-08-29 | rates 1×7 (forced full) | 2.09 s | 3.82 | 8-token verify = 2.75× single-token cost |

## Trend reading

- **Sampling width was the first win** (+36% acc, +51% t/s at top-k 20). Done.
- **Tail-trimming (n_max 4) + confidence gating (p_min 0.4) was the second**
  (+32% t/s on identical prompts). This is where we are.
- Next lever, if Tier T4/T5 is wanted: the residual golden divergence
  (~2% lattice scores, 4/7 exact path; q/k-norm eps 1e-5 vs 1e-6) and/or the
  RC-1 MoE verify-cost kernel work (F3, parked).
