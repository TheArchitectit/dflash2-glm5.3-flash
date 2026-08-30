# Proposal: add-gpu-guidance — optional GPU serving docs + harness

## Why

The project's whole framing is "CPU-only ucs03" because that's what we
measured. The user asked to make GPU support optional/available so others
can use it (`can we add optional gpu support so users can use it for that?`).
The llama.cpp dflash path is not CPU-specific — target offload (`-ngl`) and
draft offload (`-ngld`) are independent, and `--spec-draft-ngl` already
defaults to `auto`. What was missing is the guidance (flag matrix, VRAM
budget, the CPU-config-must-be-retuned caveat) and a way for a GPU user to
produce the first numbers without us pretending we have them.

We have **no GPU here** (ucs03's only display adapter is a Matrox G200
server VGA). So this change ships docs + a harness, not benchmark claims —
claiming GPU t/s would violate the repo's ship-with-raw-dumps rule.

## What Changes

- **`docs/gpu.md`** — GPU serving guide: the three recipes (full offload,
  hybrid CPU-target/GPU-draft, multi-device), dflash-relevant flag matrix
  code-verified against `common/arg.cpp` in the fork, VRAM budget for the
  1B draft, and the RC-1 reasoning for why `n_max 4 / p_min 0.4 / top-k 20`
  is CPU-specific and must be re-tuned on GPU (GPU verify batch is nearly
  free, so start from the trained `n_max 7 / p_min 0`).
- **`scripts/gpu_ab.sh`** — host-agnostic boot→health-gate→A/B→teardown
  harness with solo-run port preflight and defensive cleanup (trap on every
  exit path), reusing `ab_8prompt.py`'s corrected ruler.
- **`systemd/llama-server-glm5-dflash2-gpu.service`** — template unit (not
  installed), both offload modes documented in-comment.
- **README** — drop the "CPU-only" exclusivity framing, point at
  `docs/gpu.md`, and fix the stale "v0.0.1 in progress / HF user001" status
  line (now: shipped on GitHub, HF staged+deferred, namespace is lundrog).

## Impact

- AF affects: `docs/` (new), `scripts/`, `systemd/`, `README.md`.
- OUT: the glm5 fork (no code change — flags already exist and work; the
  only new "code" is our harness).
- OUT: Qwen :8086 (standing rule — untouched).
- OUT: any GPU benchmark claim in `research/08` or the results write-up
  (nothing is recorded there until a real card produces it).
