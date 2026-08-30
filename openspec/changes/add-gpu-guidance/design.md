# Design: add-gpu-guidance

## Constraint that shapes everything

No GPU on this box. We can verify flags against source and reason about
costs; we cannot produce numbers. So the deliverable is a **harness + guide
that a GPU user runs**, not a result table. Every number-shaped sentence in
`docs/gpu.md` is either cited from CPU measurement, marked "code-verified",
or explicitly labelled unmeasured — no exceptions. This is REQ-WF hygiene
(loop never claims an unmeasured result) applied to docs.

## Why the hybrid mode is the headline

For this specific model the interesting GPU case is not `-ngl 99` (most
cards can't hold a 147 GB IQ4_XS target). It's **`-ngl 0 -ngld all`**: the
bandwidth-bound target stays on CPU where we measured it, and the drafter —
which runs 8 forward passes per verify step and is only ~2.2 GB — moves to
GPU. That turns the draft's compute cost into near-zero without needing a
monster card. llama.cpp supports it because the draft is its own
`llama_context` with its own `n_gpu_layers` (`--spec-draft-ngl`, default
`auto`), so the two offload levels are genuinely independent.

## Why `n_max 7 / p_min 0` on GPU but `4 / 0.4` on CPU

CPU locked config exists purely because of RC-1: verify batch ~2.75× a
single token on the bandwidth-bound path, so short gated blocks win. On a
GPU that batch cost is nearly free, so the trained full block (which accepts
more tokens/step) dominates — the published 5.4–5.8 GPU numbers are exactly
this regime. `gpu_ab.sh` defaults to the GPU-appropriate block but exposes
`--n-max`/`--p-min` so the retune is one flag away and A/B-able on the same
corrected ruler.

## Harness safety

Solo-run rule survives portability: `gpu_ab.sh` health-probes the target
port first and **refuses to boot** a second server onto a busy port, rather
than racing. Cleanup is `trap ... EXIT INT TERM` with a bounded TERM→KILL
reap — the lesson from the Sprint 4.2 chain that died on a `set -e` +
"Job canceled" race (see `research/08`).
