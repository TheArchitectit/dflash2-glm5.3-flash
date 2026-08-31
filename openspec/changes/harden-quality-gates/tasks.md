# Tasks: harden-quality-gates

Protocol: detectors before detected (design "Shape"); every gate-behavior
change is demonstrated red→green in the commit message, not asserted.

## G1 — crash adjudication (applied 2026-08-31)

- [x] `run-tests.mjs`: solo re-run clears a failure only with
  `fail === 0 && code === 0 && !timedOut`; crash reason surfaced in the
  solo ✗ line. Red→green record: baseline exit 0 with `(flake)` masking
  → post-detector-fix exit 1 (`crashed, code 2` under FAILED FILES) →
  post-test-fix `1 passed, 1 skipped`, exit 0.
- [x] `test_hc_collapse.py`: lazy `gguf` import behind `GGUF_PY`, missing
  package → real-weights arm `pytest.skip`s (synthetic arms run),
  `RuntimeError` → synthetic fallback in the loader helper, loud error on
  non-F32/Q8_0 tensor types, `--target-gguf-dir` wired to the loader
  (was parsed and ignored).

## G2 — fresh-clone battery (applied 2026-08-31)

- [x] README quality-gates section: `npm install` one-time setup note;
  semantic-scan line now states the no-op truth (no TS/JS sources).

## G3 — knob honesty (applied 2026-08-31)

- [x] `bench_greedy_lossless.py`: GATE compares `match == len(a)` and
  prints `GATE n/n` from the actual run size (10 stays the default,
  stops being a constant; REQ-QG4).
- [ ] Box-side: re-run the REQ-SD-4 arms once to confirm 10/10 still
  holds under the generalized gate (needs both servers; not runnable
  off the measurement box).
- [x] `diff_gguf_meta.py`: `or True` branch condition deleted;
  MUST_MATCH comparison flattened to a plain if/else (same verdict,
  readable operator precedence).

## G4 — spec hygiene (applied 2026-08-31)

- [x] `openspec/changes/add-gpu-guidance/.openspec.yaml`: `skip_specs:
  true` (docs-only change — no delta intended).
- [x] `openspec/changes/release-v001-dynamic/specs/release-workflow/spec.md`:
  insert the `## ADDED Requirements` section header (format-only).
- [x] `openspec validate --all` exits 0 for every change (output recorded
  in the closure summary of this session).

## G5 — evidence hygiene (applied 2026-08-31)

- [x] `scripts/check_raw_dumps.py`: whole-content repetition detection
  (k ∈ {2,3}, byte- and line-level, one-line trailing tolerance for a
  final newline), exit 1 on undocumented duplication, reason-carrying
  allowlist at `benchmarks/raw/.duplication-allowlist`. Red/green demo:
  doubled scratch capture → exit 1 naming the file; real
  `benchmarks/raw/` → exit 0, all 16 dumps clean.
- [x] Correction of record: the initial QA claim that
  `acceptance_3.6_50prompt.log` was double-captured is RETRACTED — the
  file is 18 lines; head/tail window overlap produced two
  identical-looking views. Retraction recorded in proposal + design D4;
  no provenance note shipped; allowlist ships empty with format docs.
  No dump was edited.
- [x] `check_raw_dumps.py` added to the README gate battery.

## G6 — upstream note (no code here)

- [x] Upstream-ready patch packaged at
  `notes/devgate-upstream-adjudication-fix.md` (bug, diff, rationale,
  verification record; the vendored copy carries the fix since `61a88ca`).
- [ ] File the PR against TheArchitectit/DevGate-Agentic-Framework and
  link it in that note (external repo — owner's call).

## G7 — late sweep: remaining QA-report items (applied 2026-08-31)

- [x] `tests/golden/replay_dflash2.cpp`: null-check ordering — the
  `if (!ctx)` guard now precedes `llama_set_embeddings_nextn` /
  `llama_set_causal_attn` (they dereference ctx).
- [x] `scripts/gpu_ab.sh`: `grep -c` exits 1 on zero matches, which under
  `set -euo pipefail` failed a CLEAN run at its last step; captured with
  `|| true`.
- [x] `scripts/bench_gsm8k_mirror.py`: docstring states the sampling-tail
  caveat (top_k 40096-of-154880 approximates the published "no top-k";
  mirror is not an exact protocol match).
- [x] Golden-chain dead code (pyflakes): `compare_golden.py` unused
  `json` import + dead `sc`/`ss` locals; `sglang_ref_dump.py` unused
  `sys` import + dead `kv_size`/`pos_arr`.

## G8 — CI + openspec queue reconciliation (applied 2026-08-31)

- [x] `.github/workflows/gates.yml`: the full README battery + gitleaks
  (tree + history, the repo's pre-push discipline) on every push/PR —
  REQ-QG3 becomes enforced, not remembered. Runs off-box by design: npm
  ci, numpy+pytest, openspec CLI pinned @1.10.0; no /mnt assets, no gguf.
- [x] `openspec list` drift fixed: `add-dflash2-support` read 0/38 while
  v0.0.1 shipped — reconciled 35/38 against repo evidence only
  (preamble in its tasks.md records the basis; 3.7 / 4.6 / 4.7 stay open
  and honest). `release-v001-dynamic` 7/10 → 9/10 (L1/L2 RETIRED with the
  drained loop; T6 deferred-by-user remains the one open item).
