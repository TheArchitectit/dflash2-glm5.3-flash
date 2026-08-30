# Proposal: add-moe-kernel-tuning — attack RC-1 at the kernel level

## Why

v0.0.1's +40% came from config tuning alone. The remaining headroom is
RC-1 (`research/07`): an 8-token verify batch costs **2.75× a single token**
on CPU (synth-rate calibration: 2.09 s vs 0.758 s), which caps acceptance
benefit and forced our short-block lock. F3 ("expert-reuse-aware verify")
was parked as weeks-scale; this change does the day-scale part properly:
**measure where the 1.75 s of excess actually goes before writing any
kernel.**

## Design correction discovered during source review (before coding)

The original F3 framing assumed drafted tokens re-read expert weights per
token. Reading the actual kernel (`ggml/src/ggml-cpu/ggml-cpu.c:1463-1524,
1622-1637`) refutes that: `mul_mat_id` already (a) buckets token-choices by
expert, and (b) processes each bucket row-blocked (`blck_0=16`) ×
column-blocked (`blck_1=16`) — a 16-row × 16-token weight tile stays hot in
L1 while walking tokens. Weight DRAM traffic is therefore already close to
"once per touched expert" for buckets ≤16 wide. Expert *collisions* mostly
matter via **which and how many distinct experts a block touches** (cold
weight pages), not via duplicate vec_dots.

Consequence: the decisive measurement is not a collision counter — it's the
**verify cost curve**: `cost(n) = a + b·n` from synth-rate runs at n=1..8.
- flat (a≈1×) → weights already amortized; the excess is b·n (KDA recurrent
  state, KV, activations) → MoE dedup is the wrong lever;
- steep (b≈0.7+) → same conclusion, bigger;
- a >> expected weight floor → routing touches too many cold experts;
  collision/expert-count data then localizes the win.

Also verified: **IQ4_XS has no repack fast-path traits**
(`repack.cpp:4528-4551`: q4_0/q2-K/q4-K/q5-K/q6-K/iq4_nl only) — an
independent, generic lever (2b) if perf shows mul_mat_id dominant.

## What Changes

- **K-1 cost-curve experiment**: `--spec-synth-rates` at n_max 1..8 (fork
  already supports; no code change), fit a+b·n. Answers the lever question
  with wall time, zero instrumentation risk.
- **K-2 collision/expert-count probe**: env-gated (`LLAMA_MOE_PROBE=1`)
  thread-0 counters in the grouping pass (pairs, distinct experts per call,
  cne1 histogram) — cheap sanity channel on the routing floor assumption.
- **K-3 attribution**: `perf record` + `perf report` on the synth run (perf
  exists on the box) — op-class shares (mul_mat_id vs ssm/KDA vs mul_mat
  attention) without a timing-profiler build.
- **Gate → K-4**: pick ONE kernel change from the data: (a) expert-count/
  collision-driven change, (b) IQ4_XS repack traits, or (c) KDA/state-save
  batching — or retire the whole direction with the measured curve as the
  published negative result.
- Host rule relaxed by user for this phase: no production on this host —
  llama-server may be started/killed freely (still: one 147 GB instance at
  a time, RAM preflight).

## Impact

- AF: fork copy `/mnt/ollama/models/llama-cpp-kernel` (K-2 patch only, one
  clean commit, env-gated); `benchmarks/raw/`, `research/07,08`, this
  change's tasks.
- OUT: production fork tree `/mnt/ollama/models/llama-cpp-glm5` (untouched;
  :8100 may run there as a reference baseline, and the measurement windows
  that need the box must not overlap with it).
- OUT: any kernel change before the K-4 gate says which one.
