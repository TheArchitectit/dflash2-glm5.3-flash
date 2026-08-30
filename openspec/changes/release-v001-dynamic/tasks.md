# Tasks: release-v0.0.1 (dynamic queue tail)

Protocol: follow design.md (one step/tick, Monitor wake + heartbeat, verify
live state, commit per step). Escalation triggers in REQ-WF-4.

## In flight

- [ ] **T1 (4.5, part 1)** — Q8_0 + BF16 draft conversions in background
  (`/tmp/conv_variants.log`, DONE-TOKEN `ALLCONV`). Gate: files exist,
  ~2.3/1.2 GB; each passes `scripts/check_tensor_inventory.py` +
  `check_conv_base.py` + `diff_gguf_meta.py` (Sprint-1 gates, variant mode).
- [ ] **T2 (4.5, part 2)** — acceptance A/B vs F16: solo swap windows,
  `scripts/ab_8prompt.py` per variant. Gate: Δacc ≤ 2% → publish all three,
  else F16-only and document. Log DONE-TOKENs `ABQ8DONE`/`ABBFDONE`.

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
