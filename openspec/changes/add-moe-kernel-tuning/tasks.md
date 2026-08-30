# Tasks: add-moe-kernel-tuning (autonomous queue)

Protocol: design.md ladder + release-v001-dynamic loop mechanics, with the
user amendment: no production to protect — :8100 stopped only to free RAM
for a measurement window, restored at window end. One step per tick,
commit per step, DONE-token to the window log.

## K1 — spec + harness (this change)

- [x] proposal / design / tasks.
- [x] spec delta `specs/kernels/spec.md` (REQ-K1..K3).
- [x] `scripts/k1_cost_curve.sh` — boots probe-build server at
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

- [x] **W1 cost curve COMPLETE 2026-08-30 05:26** — cost(8)=1.866 s,
  cost(5)=1.391 s; fit a+b·(n−1): **a=0.759** (== single-token 0.758,
  weight traffic amortized), **b=0.158 s = 0.21×/token (FLAT)**. Kill-check:
  1.87 s at 16k ctx vs Sprint-5.1 2.09 s at 131k — linear in ctx, passes.
  Raw: `benchmarks/raw/k1_win1_n{7,4}.log`, `k1_window1.log`.
- [x] **W2 probe COMPLETE (same windows)** — 8-token verify touches **38.3
  distinct experts**/64 slots vs uncorrelated floor 57.9; cne1>8 never;
  excess-vs-n2-baseline col_rate +0.34. `moe_probe_win1.txt` +
  `scripts/moe_probe_parse.py`.
- [x] **W3 attribution COMPLETE 2026-08-30 05:52** — perf record LBR on
  synth n_max=7 decode-only (2.84M samples), `paranoid` raised 4→2 for the
  window then reverted. Top self: `iq3_s_q8_K` **32.89%**, `iq4_xs_q8_K`
  5.88%, `q6_K` 1.91%, KDA `gated_delta_net` **1.04%**, flash_attn 0.74%.
  Report: `benchmarks/raw/k3_perf_report.txt`, window `k3_window.log`.
  **Overrides the W1 branch table:** KDA batching (c) DEAD (1.04%); the
  planned IQ4_XS repack (b) was aimed at the wrong quant (IQ4_XS only 5.9%
  — this mixed target's routed experts are mostly **IQ3_S**, 82/129
  tensors); expert-reuse (a) bounded to ~8-12% (< gate, existing 16×16
  blocking already captures per-bucket reuse).

## K-4 — revised branch: mixed-IQ microkernel (IQ3_S first)

- [x] Amdahl computed: multi-token buckets = 35% of touched / 61% of pairs;
  expert-batch 1.5-3x → 1.086-1.188x; IQ-dot 1.2-2x → 1.069-1.240x. Only a
  large speedup of the hot IQ3_S dot clears ≥15%. `research/09` W3 section.
- [x] **Design pass COMPLETE 2026-08-30 06:20** (cpp-pro, read-only) — can IQ3_S /
  IQ4_XS drop into `repack.cpp` `tensor_traits<BLOC,INTER,NB_COLS>`
  templates, or need a bespoke AVX-512/VNNI microkernel? All `quants.c`
  vec_dot assert `nrc==1` (no multi-column path) — fusion must be a new
  kernel, not a flag. Determines "add traits" vs "write kernel".
- [x] K-4 writeup final — 07 SUPERSEDED banner + amendment section, 08 closure row, 09 DECISION section (bb8712a, 9dd2b70). branch + gate + `research/08` trend row +
  `research/07` RC-1 amendment (the "~6.5x weight read" was an
  overestimate; measured 4.8x touched experts, already L1-amortized).
- [x] **K-4 DECISION: RETIRED on this hardware** (no `kernel-iq3s-dot` change).
  Box is Haswell (E5-2660 v3): AVX2-only, no AVX-512 (verified /proc/cpuinfo);
  repack traits structurally incompatible with IQ3_S layout; fused multi-token
  gemv asserts nrc==1 + bandwidth-bounded (<5%); combined-Amdahl best case
  8.6% < 15% gate regardless. Portable AVX-512 follow-on drafted for community
  notes (b7cceb0). Loop drained — add-moe-kernel-tuning COMPLETE.

## Rules

- Never concurrent 147 GB loads; RAM ≥ 200 GB before any boot.
- :8100 restore is part of every window's completion criteria (L2).
- No kernel behavior change lands anywhere until K-4 names it; the probe
  commit is read-only observability.
- All measurements cross-checked against server-side telemetry before
  recording (memory: bench-formula-crosscheck-server-truth).
