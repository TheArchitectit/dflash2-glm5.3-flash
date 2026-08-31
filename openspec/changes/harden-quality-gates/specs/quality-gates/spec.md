# quality-gates — the battery must not pass silently

## ADDED Requirements

### Requirement: REQ-QG1 A crashed test file fails the gate

The test runner SHALL treat a test file whose process exits nonzero
without reporting test results (collection error, missing dependency,
signal kill) as FAILED. A solo re-run MAY clear a parallel-lane failure
as a flake only when the solo run reports zero failures AND exits 0 AND
did not time out; every other solo outcome stays failed and the gate
exits 1. Adjudication SHALL rely on process exit codes, not on parsing
test-runner output for failure counts.

#### Scenario: import error at collection

- **WHEN** a collected test file crashes on import in both the parallel and the solo run
- **THEN** the file is listed under FAILED FILES with its exit code and the gate exits 1

#### Scenario: genuine flake

- **WHEN** a file reports real test failures in the parallel lane but exits 0 with zero failures in the solo re-run
- **THEN** it is reported as flaky and the gate MAY exit 0

#### Scenario: solo re-run hangs or is killed

- **WHEN** the solo re-run times out or dies to a signal
- **THEN** the file stays failed and the gate exits 1

### Requirement: REQ-QG2 Optional heavyweight dependencies degrade to skip

Test files SHALL NOT import model-conversion or inference packages
(`gguf`, `torch`) at module level when only a subset of tests need them.
When such a package is unavailable, the tests requiring it SHALL skip
with a reason naming the env var that supplies it, and the
dependency-free tests in the same file SHALL still execute and count.

#### Scenario: fresh clone without gguf-py

- **WHEN** the battery runs `tests/golden/test_hc_collapse.py` with no importable `gguf`
- **THEN** the synthetic-semantics tests pass, the real-weights test skips with `GGUF_PY` named in the reason, and the file exits 0

### Requirement: REQ-QG3 The declared battery passes on a fresh clone

After `npm install` on a clean checkout with none of the measurement
box's `/mnt` assets present, every gate listed in the README SHALL exit
0, and `openspec validate --all` SHALL report zero failing changes.
Documentation SHALL describe each gate's actual behavior — a scan that
finds no TS/JS sources is a stated no-op — and docs-only changes SHALL
declare `skip_specs: true` rather than inventing deltas.

#### Scenario: clean machine

- **WHEN** the full documented battery plus `openspec validate --all` runs on a machine without the measurement assets
- **THEN** every gate exits 0 and validate reports no failing changes

### Requirement: REQ-QG4 Gates honor their own knobs

A benchmark or gate script exposing a CLI knob that controls run size
SHALL derive its pass threshold from that knob's value, not from a
constant. A run that satisfies the knob-scaled threshold SHALL pass.

#### Scenario: lossless gate at reduced n

- **WHEN** `bench_greedy_lossless.py --n 5` completes with 5/5 identical arm outputs
- **THEN** the GATE line prints PASS and the script exits 0

### Requirement: REQ-QG5 Raw dumps are append-only evidence

Raw benchmark dumps SHALL NOT be edited after capture. Capture artifacts
such as duplicated content from a double-invoked `tee` SHALL be detected
by a checker, recorded in the results write-up, and allowlisted with a
one-line reason — never silently normalized or rewritten.

#### Scenario: duplicated log content

- **WHEN** a raw log in `benchmarks/raw/` contains its content more than once and is not in the duplication allowlist
- **THEN** the checker exits 1 naming the file; an allowlisted file is reported with its recorded reason and the file itself remains byte-identical
