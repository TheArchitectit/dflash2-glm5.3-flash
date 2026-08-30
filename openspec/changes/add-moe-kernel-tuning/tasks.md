# Tasks: add-moe-kernel-tuning (autonomous queue)

Protocol: design.md ladder + release-v001-dynamic loop mechanics, with the
user amendment: no production to protect — :8100 stopped only to free RAM
for a measurement window, restored at window end. One step per tick,
commit per step, DONE-token to the window log.

## K1 — spec + harness (this change)

- [x] proposal / design / tasks.
- [x] spec delta `specs/kernels/spec.md` (REQ-K1..K3).
- [ ] `scripts/k1_cost_curve.sh` — boots probe-build server at
  n_max {7,4} with matching-length full-acceptance synth rates (the fork
  asserts `rates.size()==n_max`), runs `ab_8prompt.py --n-predict 64`
  ×3 reps, parses server decode telemetry, emits `RESULT cost_n=<n>
  cycle_s=<x>` lines. Defensive: trap cleanup, health-gate, never
  double-boot (port preflight).

## K2 — probe patch (fork copy)

- [ ] `LLAMA_MOE_PROBE` counters in `ggml_compute_forward_mul_mat_id`
  grouping pass (design K-2 spec): calls/pairs/touched/collisions/cne1-hist
  bucketed by n_tokens∈{1,4,8}; periodic + atexit log; zero-perturbation
  when unset (one bool branch).
- [ ] build `build-probe/` (Release, GGML_NATIVE=ON); binary
  `llama-server` smoke: health + `draft_n>0` + probe lines appear with env,
  absent without.
- [ ] one commit on branch `kernel/moe-probe` in llama-cpp-kernel (clean
  revert path).

## K3 — windows (each: stop :8100 → preflight → run → kill → restore → commit)

- [ ] W1 cost curve: n_max=8 + synth 1×7 → reproduce 2.09 s ±10% kill-check
  → n_max=4 + synth 1×3. Raw: `benchmarks/raw/k1_cost_curve.log`.
- [ ] W2 probe: same synth run with `LLAMA_MOE_PROBE=1`; journal →
  `benchmarks/raw/moe_probe.json` (`scripts/moe_probe_parse.py`).
- [ ] W3 attribution: `perf record` (lbr) during synth n=8; `perf report`
  top-20 → `benchmarks/raw/k3_perf.txt`.
- [ ] K-4 gate: `research/09-kernel-findings.md` (fit, shares, histogram,
  branch chosen) + `research/07` RC-1 amendment + `research/08` trend rows.
- [ ] If branch chosen: open `kernel-<lever>` openspec change; else record
  retirement decision here and stop loop.

## Rules

- Never concurrent 147 GB loads; RAM ≥ 200 GB before any boot.
- :8100 restore is part of every window's completion criteria (L2).
- No kernel behavior change lands anywhere until K-4 names it; the probe
  commit is read-only observability.
- All measurements cross-checked against server-side telemetry before
  recording (memory: bench-formula-crosscheck-server-truth).
