# Tasks: release-v0.0.1 (dynamic queue tail)

Protocol: follow design.md (one step/tick, Monitor wake + heartbeat, verify
live state, commit per step). Escalation triggers in REQ-WF-4.

## In flight

- [x] **T1 (4.5, part 1) COMPLETE 2026-08-30** — q8_0 (1.25 GB) + bf16
  (2.35 GB) converted at `/mnt/ollama/models/glm-5.3-flash/dflash2-gguf/`;
  both PASS check_tensor_inventory + check_conv_base; q8_0-vs-f16
  diff_gguf_meta PASSED (allowlisted-only). Note: diff_gguf_meta exit verified 0 on clean runs (the earlier
  "exits 1" note was a misread of a piped command's exit status — corrected
  by direct repro at T2 close).
- [x] **T2 (4.5, part 2) COMPLETE 2026-08-30** — solo swap chain, both
  variants A/B'd vs the F16 re-baseline on the corrected ruler (8-prompt,
  n_predict 32): F16 1.79/1.65, **Q8_0 1.85/1.80**, **BF16 1.99/1.92**.
  Neither variant DEGRADES acceptance (Δ +3.4%/+11.2% — selector precision
  survives quantization; spread is run noise at n=8). Gate reads "no
  degradation → publish all three": **publish F16 + Q8_0 + BF16**. Production
  unit verified restored to f16 (unit file + live journal + health).

## Close-out chain (pre-authorized; gate = scan clean)

- [ ] **T3** — DevGate full pass on final commit: `guardrails-scan`,
  `regression_check --all --pre-commit`, `run-tests`; dynamic timeouts wired
  per test (carry-over request). Gate: all green.
- [ ] **T4** — secret scan at release commit: tree + `git log -p --all` +
  `git ls-tree` large-blob audit. Gate: zero findings any tier. Any finding
  → STOP + surface (REQ-WF-4).
- [ ] **T5** — create public repo (name: dflash2-llamacpp; decide org vs user001
  namespace; LICENSE MIT for code + note draft weights cc-by-nc-nd-4.0), push,
  then flip private→public. Gate: repo 200s, CI/Actions off, README renders.
- [ ] **T6 (4.6)** — HF repo `user001/GLM-5.3-Flash-DFlash2-GGUF`: F16 (+Q8_0
  if T2 passed), README = results-dflash2-glm.md numbers + license
  cc-by-nc-nd-4.0 (weights) + base_model: incoai/GLM-5.3-Flash-DFlash2 +
  serving recipe verified by curl against live :8100. Gate: download-back
  smoke + check_tensor_inventory on the re-fetched file.
- [ ] **T7 (4.7)** — notes: z-lab/dflash discussion + llama.cpp note (mHC
  capture semantics finding, n_max/p_min levers, tiered numbers, CPU context).
  Gate: links recorded in notes.md.
- [ ] **T8 (4.8)** — tag `v1.0.0-dflash2-glm`, README status checklist,
  memory update, OpenSpec change marked complete, loop STOPS (empty queue).

## Loop hygiene

- [ ] **L1** — every tick: verify state from ground truth (log exists, units,
  ports) — never prior-tick memory (chain-death incident 2026-08-29/30).
- [ ] **L2** — restore :8100 on every failure path of any swap chain.
