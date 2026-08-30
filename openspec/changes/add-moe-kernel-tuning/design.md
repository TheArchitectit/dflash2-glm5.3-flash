# Design: add-moe-kernel-tuning

## Shape: measure → attribute → one kernel change → ship or retire

Autonomous /loop protocol (reuses `release-v001-dynamic/design.md`
mechanics with one amendment from the user: **no production on this host —
server processes are ours to start/kill freely; the serial rule that
survives is one 147 GB instance at a time + RAM preflight ≥ 200 GB** — note
:8100 currently runs with 97 GB free, so any window starts with
`systemctl stop llama-server-glm5-dflash2` and ends by `start`).

## Experiment ladder

### K-1 — verify cost curve (no fork change; runs on the existing build)

`llama-server` at production flags + forced full acceptance. **Constraint
verified in source (`common/speculative.cpp:2435-2439`):
`--spec-synth-rates` must contain EXACTLY `n_max` non-increasing values —
so curve points come from n_max=7 with `1,1,1,1,1,1,1` (7 draft + 1 target
= the 8-token verify batch of RC-1) and n_max=4 with `1,1,1,1`:**
- Boot A: n_max=7, synth 7×1 → cost(8-token verify) directly.
- Boot B: n_max=4, synth 4×1 → cost(5-token verify).
- cost(1): spec-off single token, 0.758 s reference already measured
  (Sprint 5.1) — reuse.
Two points + origin anchor give the a/b fit; if curvature matters later,
add n_max {2,3,6} boots (rates count must match). Wall time from the
server's own decode telemetry + client t/s; ~40 min per boot window.

Kill-check baked in: cost(n) must reproduce Sprint 5.1's 2.09 s at n=8
(±10%) or the ruler/teardown protocol is wrong — investigate before
trusting the fit.

### K-2 — routing probe (fork copy, one commit, env-gated)

`LLAMA_MOE_PROBE=1`: in the thread-0 grouping pass
(`ggml-cpu.c:1622-1637`), accumulate: calls, pairs (`Σ cne1`),
`n_touched` (# experts with cne1>0), `collisions = pairs − n_touched`,
cne1-histogram (cne1∈{1,2,3,4-8,>8}), bucketed by `n_tokens` (ids->ne[1])
for n∈{1,4,8}. All plain int64s, no atomics needed (thread-0 only).
Every 5000 calls + at process exit: `GGML_LOG_INFO("[moe-probe] ...")`.
When env unset: one static-bool branch per call — effectively free, and
the counters themselves never execute.

Runs on llama-cpp-kernel build (fresh build, Release+Native, mirrors
production flags). Same windows as K-1 (probe rides along in boot C+).

### K-3 — attribution (zero code change)

`perf record -g --call-graph=dwarf -F 99 -- llama-server ...` during a
synth n=8 run; `perf report --percent-limit 0.5` → shares for
`ggml_compute_forward_mul_mat_id`, `vec_dot` family, `ggml_compute_forward_ssm_scan`,
`mul_mat`, memcpy/scatter. Cross-check against K-1's b·n: if MoE's perf
share × cost(8) ≈ the collision-explained part, levers (a)/(b) are live;
if ssm/KDA share dominates → (c).

### K-4 — the gate (human-visible decision point, still autonomous)

Write `research/09-kernel-findings.md` with cost(n) fit + shares + probe
histogram, then mechanically:
- `b`-term ≥ 0.3×single-token per extra token AND MoE perf-share ≥ 35% →
  branch (a): expert-count reduction (selector side) or (b) repack;
- MoE share ≥ 35% but flat curve → branch (b) IQ4_XS repack traits
  (helps dense prefill + verify alike);
- KDA/ssm share ≥ 30% → branch (c): recurrent-state batched update
  (graph-level: token-parallel scan for the KDA op on ≤8-token blocks);
- none above → retire F3 with the measured curve as the artifact (the
  2.75× was bandwidth, not waste — publish that in `research/08`).

Each branch then gets its own openspec change (kernel-moe-reuse /
kernel-iq4xs-repack / kernel-kda-batch) with its own A/B gate: ≥15% t/s at
locked config with golden 1e-3 + greedy lossless checks PASS, or revert.

## Solo-run windows

Same protocol as v0.0.1: stop :8100 → preflight (`free -g` ≥ 200) → launch
probe/synth config → bench → kill → restore :8100 → commit per step →
DONE-token to window log + Monitor wake. Loop never leaves a server
running between ticks.

## Risks

- 56 GB page cache < 147 GB model → cold-ish loads even on warm boots
  (~15-20 min each) — budget 3 windows, not 3 hours.
- perf dwarf unwinding on a 40-thread process is heavy: use
  `--call-graph=lbr` fallback; attribution without call graphs is enough
  (symbols are flat and named).
- Probe build divergence: llama-cpp-kernel is a tree-copy at 77445b3 —
  identical arch/code to production; re-diff before the first window.
- mlock load with probe build: same flags; if load-mode mlock pushes RAM
  preflight failures, `--load-mode normal` is measurement-safe (decode
  bandwidth unchanged after first pass; note the swap in the log).
