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

## Task #32 — GSM8K acceptance mirror vs published GPU table (2026-08-29, measured)

Methodology mirror of brandonmusic/GLM-5.3-Flash-tr3-4bpw
`runtime-results/v84/quality/gsm8k-distinct5-...json`: 5 distinct GSM8K
test rows, temp 1.0 / top-p 0.95 / no top-k / n_predict 512, DFlash2
n_max 4 + p_min 0.4 (locked config), 2 reps. Raw:
`benchmarks/raw/gsm8k_mirror.json`.

| metric | CPU llama.cpp (ours) | GPU vLLM/EXL3 (published) |
|---|---|---|
| mean acceptance | **2.693** | 5.428 |
| token-weighted | 2.632 | 5.441 |
| per-row | [2.61,2.34,2.93,2.29,3.10,2.74,2.53,2.83,2.49,3.08] | [5.25,4.89,5.27,6.03,5.70] |
| mean t/s | **2.161** (min 1.90 / max 2.48) | 145+ (different hardware) |

**Tiers:** acceptance T2 WIN; throughput **T4 STRETCH** — this run clears the
+40% ask (1.32 → 2.161 = +64%) on long-output workloads.

**Key finding — acceptance is workload-INDEPENDENT on our box:** GSM8K 2.69
vs 50-prompt agentic 2.76. The gap to the GPU's 5.43 is therefore NOT a
prompt-class artifact; it tracks the target's quantization/precision
(IQ4_XS logits vs their EXL3-4bpw/FP8 KV path) — lower-precision logits
flatten the draft-vs-target agreement, and the acceptance metric measures
exactly that agreement. Publish story: tiered numbers per workload class,
with the external GPU reference cited for lineage parity.

## Task #31 — draft-mtp A/B: NOT POSSIBLE (2026-08-29, measured)

The target ships nextn weights (`glm5next.nextn_predict_layers=[1]`) and the
fork's speculative impl accepts `--spec-type draft-mtp`, but glm5next.cpp:690
hard-asserts: `"glm5next NextN graph not implemented yet"` — crash-loop on
start (7 core-dumps before teardown; production restored, solo rule held).
MTP graphs exist for step35/qwen35/deepseek32/deepseek2/glm-dsa targets, not
this arch. Implementing the glm5next NextN graph (enorm/hnorm/eh_proj + KDA
state for the nextn block) is a substantial fork-side feature — deferred, out
of v0.0.1 scope. **DFlash2 remains the only working drafter for this target on
this box.**

## Corrected-ruler re-baseline, 8-prompt suite (2026-08-29, task #31 fallout)

Same 8 prompts / n_predict 32 as the original A/B, formula fixed
(steps = predicted_n - accepted):

| config | acc_len (old formula) | acc_len (corrected) | t/s |
|---|---|---|---|
| n_max 7 baseline | 2.38 | (not re-run; old figure mildly inflated) | 1.09 |
| **n_max 4 + p_min 0.4 (locked)** | 3.91 (inflated) | **1.79** | **1.65** |

The acceptance delta from the locked config was formula artifact (dn/7 under
p_min gating); the **t/s win is real** (+51% same-suite). Cross-check: 50-prompt
server-side ground truth 2.76 and GSM8K 2.69 are higher than this 1.79 because
n_predict 32 under-weights accepted tails vs amortized first-step overhead —
suites are not interchangeable; each number is tagged with its ruler + suite.

## Task 4.5 — draft quant variants A/B (2026-08-30, measured)

8-prompt corrected-ruler suite (n_predict 32), solo swap chains:

| draft | acc_len | t/s | verdict |
|---|---|---|---|
| F16 (production) | 1.79 | 1.65 | baseline |
| Q8_0 (1.25 GB) | **1.85** | 1.80 | no degradation (+3.4% acc, noise-range tps) |
| BF16 (2.35 GB) | **1.99** | 1.92 | no degradation (+11.2% acc, n=8 spread caveat) |

Gate (4.5): "if acceptance drops >2% → F16 only" — no drop in either
variant -> publish all three. F8 selector-precision risk closed.

## K-4 kernel-attribution result — direction selected, no throughput change yet (2026-08-30)

Measurement program `add-moe-kernel-tuning` (W1 cost curve, W2 routing probe,
W3 perf attribution). **No t/s row** — this is a research gate, not a config
change; production stays locked at n_max 4 + p_min 0.4 + top-k 20.

| window | finding | branch decision |
|---|---|---|
| W1 | verify cost FLAT: b=0.21×/token, a==single-token (weights L1-amortized) | (a) expert-reuse bounded < gate → **retire F3** |
| W2 | 8-tok verify touches 38.3 experts (not 58); correlation strong | confirms RC-1 overestimate |
| W3 | `iq3_s_q8_K` 32.9% self (hottest fn), `iq4_xs_q8_K` 5.9%, KDA 1.04% | (c) KDA batching → **DEAD**; (b) repack was aimed at wrong quant |

**K-4 outcome:** the surviving lever is a faster **IQ3_S dot microkernel**
(branch b′) — workload-agnostic, clears ≥15% only if it hits ~1.2–2× on the
dequant ALU. Follow-on: `kernel-iq3s-dot` openspec change (microbench-first
ladder before any 147 GB A/B window). Recorded as the honest end of the
measure-first phase: the remaining headroom is dequant instruction efficiency,
not routing or attention.

### K-4 CLOSURE (2026-08-30, same day) — retired on this hardware

The design pass returned and one hardware check settled it: **ucs03 is a
Xeon E5-2660 v3 (Haswell) — AVX2 only, no AVX-512/VNNI** (/proc/cpuinfo,
verified). Every surviving path to ≥15% closes: repack traits for IQ3_S/IQ4_XS
require a from-scratch `make_block_*`/`gemv_*` family (none exist; IQ3_S
layout incompatible with the QK_K-aligned pattern); fused multi-token gemv is
both a full rewrite (`assert(nrc==1); UNUSED` in every impl) *and* bandwidth-
bounded (<5%); the AVX-512 IQ3_S rewrite — the only "borderline" candidate —
has no ISA to run on here. Agent's combined-Amdahl best case: **8.6% e2e**,
under the gate regardless.

Net: **no throughput change from the kernel program** — its value is the
measurement record (flat verify cost curve, 38.3/64 correlated experts,
IQ3_S-dequant dominance) and the correction of RC-1's "~6.5× weight-read"
estimate. Portable follow-on (AVX-512 VNNI IQ3_S path, absent from the tree)
drafted in `notes/community-drafts.md` for a newer-CPU box. No kernel patch
merged; probe commit stays observability-only on `llama-cpp-kernel`
branch `kernel/moe-probe`.
