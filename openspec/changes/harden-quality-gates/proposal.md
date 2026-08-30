# Proposal: harden-quality-gates — the battery must not pass silently

## Why

An external QA pass (2026-08-31) found the repo's declared gate battery
exiting green while the only Python test in the tree never executes
anywhere. The chain:

1. `tests/golden/test_hc_collapse.py` imported `gguf` at module level
   behind a hardcoded `/mnt/ollama/.../gguf-py` sys.path — it crashes on
   any machine but the measurement box, despite its docstring claiming
   "CI passes without the 147 GB shards".
2. `.devgate/scripts/run-tests.mjs` solo adjudication cleared any failure
   with `fail === 0` — and a crashed file (pytest collection error, exit 2,
   zero tests run) emits no "N failed" lines, so a crash parses as zero
   failures and got labeled `(flake)`. Gate exit 0. Silent success in the
   very framework the repo vendors to prevent silent success.

The same family holds four more truthfulness defects: the battery breaks
on a fresh clone (`semantic-scan` needs an undocumented `npm install`;
the README claimed "skips: none" for a scan that is a complete no-op —
no TS/JS sources exist), `openspec validate --all` fails on two shipped
changes, `bench_greedy_lossless.py` hardcodes its pass threshold at 10 so
`--n 5` can never pass, and `diff_gguf_meta.py` carries a dead `or True`
branch in its allowlist logic. (An earlier draft of the QA pass also
claimed the shipped `acceptance_3.6_50prompt.log` contained its content
twice — RETRACTED on inspection: the file is 18 lines and head/tail
window overlap made two views look like two copies. The checker built
below exists so the next such claim is settled by a tool, not a diff of
windowed views.)

## What Changes

- **Crash adjudication** (applied 2026-08-31): a solo re-run clears a
  parallel-lane failure only with zero failures AND a clean exit AND no
  timeout; crashed files stay in FAILED FILES and the gate exits 1.
- **Optional-dep tolerance** (applied 2026-08-31): `gguf` imports lazily
  in `test_hc_collapse.py`; a missing package degrades the real-weights
  arm to `pytest.skip` (naming `GGUF_PY`) while the synthetic arms still
  run; non-F32/Q8_0 tensors fail loudly instead of being dequantized with
  the wrong block layout.
- **Fresh-clone battery** (applied 2026-08-31): README documents the
  `npm install` one-time setup and describes the semantic scan as the
  no-op it is.
- **Knob honesty** (pending): `bench_greedy_lossless.py` derives its GATE
  threshold from `--n`.
- **Dead-branch removal** (pending): `diff_gguf_meta.py` drops the
  `or True` leftover.
- **Spec hygiene** (pending): `openspec validate --all` green —
  `add-gpu-guidance` declares `skip_specs: true` (docs-only change);
  `release-v001-dynamic`'s delta gets its `## ADDED Requirements` header.
- **Evidence hygiene** (pending): raw dumps stay append-only evidence; a
  duplication checker flags double-captured artifacts, with a reason-
  carrying allowlist as the escape hatch. No dump currently trips it
  (see retraction above); the checker ships so the invariant is enforced
  going forward. Raw files are never edited.

## Impact

- AF: `.devgate/scripts/run-tests.mjs`, `tests/golden/test_hc_collapse.py`,
  `scripts/bench_greedy_lossless.py`, `scripts/diff_gguf_meta.py`,
  `README.md`, new `scripts/check_raw_dumps.py` +
  `benchmarks/raw/.duplication-allowlist`, format-only touches to the two
  pre-existing change folders.
- OUT: raw benchmark dumps (never edited — they are the evidence behind
  the README's claims), the vendored DevGate framework beyond
  `run-tests.mjs` (the adjudication fix is upstream-worthy; that repo is
  separate), any new measurement or benchmark claim.
