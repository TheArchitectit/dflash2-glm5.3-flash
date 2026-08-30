# Tasks: add-gpu-guidance

## Done

- [x] **G1** `docs/gpu.md` — recipes + flag matrix (each row verified against
  `common/arg.cpp` line refs in the fork) + VRAM budget + RC-1 retune caveat
  + "No CUDA/ROCm hardware here" honesty caveat.
- [x] **G2** `scripts/gpu_ab.sh` — boot/health/A-B/teardown; `bash -n` clean;
  solo-run port preflight (refuses a busy port); defensive `trap cleanup EXIT`
  so a mid-run death still reaps the child.
- [x] **G3** `systemd/llama-server-glm5-dflash2-gpu.service` — template, both
  modes in-comment, default to the trained GPU block (n_max 7 / p_min 0).
- [x] **G4** README — framing de-"CPU-only"-exclusive, `docs/gpu.md` in the
  repo map + Why-CPU section, status line corrected (shipped; HF deferred).

## Gate before commit

- [x] **G5** DevGate trio on the diff (guardrails-scan, regression_check
  --all --pre-commit, run-tests) + `bash -n gpu_ab.sh` + unit `--verify`-style
  sanity (systemd-analyze verify if systemd available; else flag-comment check).
