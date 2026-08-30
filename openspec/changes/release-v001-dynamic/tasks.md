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

- [x] **T3 COMPLETE 2026-08-30** — all three gates green on HEAD:
  `guardrails-scan` clean; `regression_check --all --pre-commit` clean
  (incl. file-size gates); `run-tests` **fixed at source**:
  test_hc_collapse.py had zero pytest functions (exit 5, mislabeled "flake")
  — reworked into 2 real tests asserting the CORRECTED capture semantics
  (T1 build_hc_mean == hc_contract unweighted mean; T2 gated contraction
  must diverge, with real learned weights when the target is present and
  synthetic fallback + skip guard). run-tests now 2 pass / 0 fail, 16.5s
  against the 120s/file cap (the dynamic-timeout carry-over: cap per file is
  wired and ample; no test needs >30s today).
- [x] **T4 COMPLETE 2026-08-30** — gitleaks v8.21.2 on tree + full history:
  "no leaks found" (32 commits, report /tmp/gitleaks_report.json). Large-blob
  audit: zero blobs >5 MB EVER in history; pack 219 KiB; largest tracked file
  69 KB (benchmarks JSONL). Regex belt (hf_/sk-/ghp_/AKIA/private-key/slack/
  google patterns) clean tree + `git log -p --all`. Zero findings all tiers →
  close-out authorized.
- [x] **T5 COMPLETE 2026-08-30** — https://github.com/TheArchitectit/dflash2-
  llamacpp (gh-auth account; user001 is the HF-side namespace, unused on
  GitHub). Created private -> pushed -> verified (no workflows, zero model
  binaries, 95 files / 608 KB, LICENSE MIT with weights carve-out) -> flipped
  public. HTTP 200.
- [ ] **T6 (4.6) — DEFERRED by user 2026-08-30** ("use github until we want
  to publish to huggingface"). GitHub repo is the working venue; HF publish
  staged and ready: `huggingface/model-card.md` (license/base_model/bench +
  recipe) + `huggingface/upload.sh` (gates baked, creates `lundrog/
  GLM-5.3-Flash-DFlash2-GGUF` — token account is `lundrog`, NOT user001;
  download-back smoke; flip public). One command to fire when the user asks.
- [x] **T7 COMPLETE (drafts) 2026-08-30** — `notes/community-drafts.md`:
  z-lab/dflash + llama.cpp discussion notes drafted with the four
  upstream-worthy findings. POSTING LEFT TO USER (outward-facing; the loop
  does not fire issues/threads on other people's repos unasked).
- [x] **T8 COMPLETE 2026-08-30** — GitHub release + tag
  `v0.0.1-dflash2-glm` (renamed from planned v1.0.0 to match the user's
  v0.0.1 framing): https://github.com/TheArchitectit/dflash2-llamacpp/releases/tag/v0.0.1-dflash2-glm.
  README status already current; memory updated. OpenSpec: T1-T5 done, T6
  deferred (staged), T7 = community posts left to the user (outward-facing).

## Loop hygiene

- [ ] **L1** — every tick: verify state from ground truth (log exists, units,
  ports) — never prior-tick memory (chain-death incident 2026-08-29/30).
- [ ] **L2** — restore :8100 on every failure path of any swap chain.
