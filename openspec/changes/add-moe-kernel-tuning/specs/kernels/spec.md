# kernels — CPU verify-cost measurement and reduction

## ADDED Requirements

### Requirement: REQ-K1 Verify cost must be measured before kernel changes

Any CPU MoE/verify kernel optimization for DFlash2 SHALL be preceded by a
measured verify-cost curve `cost(n) = a + b·n` derived from forced-full-
acceptance synth-rate runs at n_max ∈ {7,4} (rates count == n_max, verified
required by `common/speculative.cpp:2435`) with the spec-off single-token
reference (0.758 s) as the origin anchor.

The measurement SHALL reproduce the Sprint-5.1 calibration point (cost(8) ≈
2.09 s) within ±10% before the fit is trusted. No kernel change is authorized
until the fit and an op-class attribution (`perf report`) identify which of
{MoE weight traffic, IQ4_XS dequant, KDA recurrent-state} dominates the b·n
term.

#### Scenario: cost curve flat

- **WHEN** the fitted `b` is below 0.3× the single-token cost per extra
  position AND MoE's `perf` share is below 35%
- **THEN** expert-reuse (F3) is retired as the wrong lever; the measured
  curve is recorded as the negative result in `research/08`; only a lever
  matching the dominant op class may proceed

#### Scenario: MoE dominates and curve is steep

- **WHEN** `b` ≥ 0.3× per extra position AND `mul_mat_id`/`vec_dot` perf
  share ≥ 35%
- **THEN** a kernel change is opened (its own openspec change) targeting the
  MoE verify path, gated at ≥ 15% t/s at the locked config with REQ-SD-2
  golden 1e-3 and REQ-SD-4 distribution-losslessness both holding

### Requirement: REQ-K2 Probe instrumentation must be observation-only

The routing probe (expert pairs / distinct-experts / collisions / cne1
histogram bucketed by n_tokens) SHALL be env-gated (`LLAMA_MOE_PROBE=1`),
accumulate with plain integer counters in the thread-0 grouping pass only
(no atomics, no per-token work), and impose no measurable overhead when the
env is unset (single static bool branch). It SHALL NOT alter decode
behavior, output, or timing; a run with the probe built but the env unset
MUST be byte-identical to the same run without the patch.

#### Scenario: probe perturbs timing

- **WHEN** a probe build with the env unset differs from the baseline build
  by more than measurement noise (> 2% t/s, self-determinism intact)
- **THEN** the instrumentation is considered broken and must be redesigned
  before any cost attribution uses it

### Requirement: REQ-K3 Kernel changes preserve losslessness and self-determinism

Any kernel change selected by K-4 SHALL pass `scripts/bench_greedy_lossless.py`
(both arms self-deterministic; distribution-level equality) and
`tests/golden/` (draft hiddens within 1e-3) before it can be benchmarked for
the ≥15% gate. A batched/accumulated kernel MAY flip bitwise near-ties versus
the current GEMV-order path (accumulation order changes) — that is within
REQ-SD-4's accepted scope and SHALL NOT be treated as a correctness failure,
provided structural divergence stays at the tie level.

#### Scenario: batched kernel flips a tie

- **WHEN** the new kernel produces a different token than the current build
  at a near-tie position, but both builds are each self-deterministic and
  distribution-level checks pass
- **THEN** proceed; record the divergence count in `research/08` as expected
  numerical-order behavior, not a regression
