# Improvement Tracking — DFlash2 CPU throughput over time

Every measured config lands here exactly once, newest row on top. Tiers are
derived from the ladder in `benchmarks/acceptance-gate.md` — never hand-edited.

## 8-prompt A/B suite (scripts/bench_acceptance.py defaults, /tmp/ab_bench.py)

| Date | Config | acc_len | t/s | Tier(t/s) | Tier(acc) | Δ vs prev |
|---|---|---|---|---|---|---|
| 2026-08-29 | n_max 4 + p_min 0.4 + top-k 20 | 3.91 (inflated by dn/7 — real acc closer to baseline; t/s row below stands) | 1.44 | T2 WIN | — | **+32% t/s** |
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
  (+32% t/s on identical prompts — the honest headline is the throughput; the
  acceptance row for n_max=4 was inflated by the buggy client formula). This is where we are.
- Next lever, if Tier T4/T5 is wanted: the residual golden divergence
  (~2% lattice scores, 4/7 exact path; q/k-norm eps 1e-5 vs 1e-6) and/or the
  RC-1 MoE verify-cost kernel work (F3, parked).

## Sprint 3.6 — full 50-prompt acceptance gate (2026-08-29, measured)

Config: n_max 4 + p_min 0.4 + top-k 20 (locked). Raw log:
`benchmarks/raw/acceptance_3.6_50prompt.log`.

| Date | Config | prompts | acc_len | t/s | Tier(t/s) | Tier(acc) |
|---|---|---|---|---|---|---|
| 2026-08-29 | n_max 4 + p_min 0.4 + top-k 20 | 50 | **2.76 (corrected; client formula was dn/7)** | **1.864** | **T3 TARGET (+41% vs 1.32)** | **T2 WIN** |

> **Formula correction (same day):** `scripts/bench_acceptance.py` computed
> `steps = draft_n / 7`, valid only for full n_max=7 blocks — and the locked
> config drafts ≤4 under p_min gating. Server-side ground truth from journalctl
> (`mean len`, computed exactly as server-context.cpp:664 does) over the 3.6
> window: **mean acceptance 2.76** (range 1.56–4.23). The client formula is now
> fixed to `steps = predicted_n - accepted`, which matches the server identity.
> t/s figures were never affected (straight from predicted_per_second). The
> 8-prompt A/B acceptance row below is likewise inflated for n_max=4; its t/s
> row stands.

Old hard gate (≥5.0) reads FAIL, but per benchmarks/acceptance-gate.md the
binding constraints are: correctness golden (PASSED, 1e-3), lossless check
(pending 3.7), and T0 net-loss (cleanly cleared — 1.864 >> 1.32 baseline).

## Sprint 3.7 — greedy lossless check: 2/10 IDENTICAL (2026-08-29)

| arm | server | prompts | identical |
|---|---|---|---|
| A (spec on, :8100) vs B (spec off, :8101) | serial solo runs | 10 | **2/10** |
| B rerun vs B (spec off self-consistency) | same unit | 3 | 3/3 DETERMINISTIC |

**Reading:** spec-off greedy is fully self-deterministic, so the 8 divergences
are not box/thread noise. They are spec-path-specific. Accept logic audited
(common/sampling.cpp:678): the target SAMPLES its own distribution at each
verify position and accepts only on exact draft match — distributionally
lossless for any sampler (temp 0 included), but NOT bitwise-lossless. For
greedy to match arm-for-arm, the 5-token verify batch (GEMM) would have to be
bit-identical to single-token decode (GEMV) — on CPU the reduction order
differs and near-tied argmaxes flip (signature matches: "ReplicaSets/Pods"
casing, mid-stream drift, identical opening tokens). The 2 passing prompts are
the high-confidence ones (bash one-liner, prime list).

**Consequence:** REQ-SD-4 as written ("identical greedy outputs") cannot pass
on a CPU batched verify. Options: (a) redefine lossless as distribution-level
(the provably correct claim for spec decode) + add a self-consistency gate;
(b) chase bitwise via single-token verify recompute (kills the speed win);
(c) keep the hard gate and document the CPU caveat. Pending decision; the
correctness tier (REQ-SD-2 golden, 1e-3) is unaffected and still PASSED.
