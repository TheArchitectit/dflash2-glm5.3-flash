# Design: harden-quality-gates

## Shape: fix the detector, then the detected

The applied work followed a strict order on purpose: fix `run-tests.mjs`
FIRST, re-run (gate goes red on unchanged code — proof the masking was
real), then fix the test (gate goes green with a genuine pass). Fixing
the test first would have produced the same end state with no evidence
the detector ever worked. Future gate changes keep that discipline:
detectors before detected.

## D1 — Adjudication trusts exit codes, not output parsing

`soloClean = solo.fail === 0 && solo.code === 0 && !solo.timedOut`.

`code === 0` is the load-bearing term. Exit codes are contractual:
pytest exits 2 on interrupted collection, 1 on test failures, 0 on pass;
node's test runner exits 0 on pass; a signal kill resolves to `null`,
which `!== 0`. The original bug existed because `fail` was derived by
regexing "N failed" lines out of stdout — a crash prints none, so `fail`
parsed as 0. Deepening that regex (e.g. also matching "ERROR") would
repeat the mistake against the next output-format change. Any future
runner (pytest -> anything) only has to keep its exit-code contract.

Genuine flakes still adjudicate cleanly: real failures in the parallel
lane, clean solo pass (exit 0, zero fails) → flake, gate may exit 0.
Timeouts and signal kills in solo stay failed — a file that only passes
when nothing else competes for CPU is a resource bug, not a flake, on a
box with a solo-run rule.

## D2 — Optional deps skip, they don't crash collection

pytest semantics: a module-level `ImportError` kills the whole file at
collection — including the tests that never needed the package. The fix
pattern: try-import at module level, `gguf = None` on failure, the
real-weights test `pytest.skip`s with the env var named in the reason,
synthetic-arm tests run regardless. One test file then has three
honest states (pass / skip-with-reason / fail) instead of one dishonest
one (crash → masked).

The Q8_0 guard in the same loader is the same principle at the tensor
level: the old code dequantized ANY non-F32 type with the Q8_0 block
layout, and the shape assert below it could still pass on a wrong-type
tensor (block counts coincide across quant types). Wrong layout must
fail loudly (`RuntimeError` naming the tensor and type), because silent
garbage weights would corrupt exactly the capture-semantics check the
file exists to protect.

## D3 — `skip_specs` for docs-only changes, not invented deltas

`add-gpu-guidance` intentionally modifies no requirement — inventing a
delta to satisfy the validator would be spec theater. The tool's escape
hatch (`skip_specs: true` in the change's `.openspec.yaml`) is the
honest declaration. `release-v001-dynamic` HAS requirements; its spec
file was just missing the `## ADDED Requirements` section header — a
format fix, no semantic change.

## D4 — Raw dumps are evidence, so the checker documents, never edits

The repo's contract is "benchmark claims ship with their raw dumps."
The checker therefore: detects whole-content repetition (k ∈ {2,3}
identical byte or line chunks, one-line trailing tolerance for a final
newline captured in only some passes), exits 1 on undocumented
duplications, and consults `benchmarks/raw/.duplication-allowlist` where
each entry carries a one-line reason. It ships empty — the initial QA
pass's claim that `acceptance_3.6_50prompt.log` was double-captured did
not survive inspection (18-line file; head/tail window overlap produced
two identical-looking views). Retracting it in the change texts, rather
than quietly dropping it, is the same discipline REQ-QG5 encodes: claims
about evidence get settled by a tool and a written record, not by
editing or forgetting.

## D5 — Knobs and dead code

`bench_greedy_lossless.py --n 5` with 5/5 matches must PASS: the gate
compares `match == len(a) == args.n`. The `== 10` constant was a
leftover from REQ-SD-4's fixed 10-prompt set; the flag existed but was
decorative. `diff_gguf_meta.py`'s `if f.name == "..." or True:` is a
debug leftover that makes the condition meaningless — remove it and, in
the same touch, flatten the unreadable nested ternary on the MUST_MATCH
comparison so the next reader doesn't have to re-derive operator
precedence to trust a STOP GATE.
