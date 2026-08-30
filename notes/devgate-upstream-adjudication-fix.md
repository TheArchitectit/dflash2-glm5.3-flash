# DevGate upstream: run-tests.mjs crashed-file adjudication fix

Prepared 2026-08-31 (openspec `harden-quality-gates` G6). The vendored
DevGate copy in this repo carries the fix (commit `61a88ca`); this note
packages it for filing against
[TheArchitectit/DevGate-Agentic-Framework](https://github.com/TheArchitectit/DevGate-Agentic-Framework)
upstream. Filing the PR is the repo owner's call (external repo).

## Bug

`.devgate/scripts/run-tests.mjs`, solo adjudication:

```js
if (solo.fail === 0) {   // ← the bug
```

`solo.fail` is parsed from the runner's output ("N failed" / "not ok"
lines). A test file that **crashes before running any test** — pytest
collection error, module-level ImportError — emits no such lines, so
`fail` parses as 0. Result: the crashed file is adjudicated as a flake,
the gate exits **0**, and the whole suite can be silently dead (this
repo's only Python test never ran anywhere, and the gate stayed green).
A silent-success failure mode in the framework whose job is preventing
silent success.

## Fix (exit codes, not output parsing)

```diff
--- a/scripts/run-tests.mjs
+++ b/scripts/run-tests.mjs
@@
 			console.error(`▶ solo: ${r.file}`);
 			const solo = await runOne(join(PROJECT_ROOT, r.file));
-			if (solo.fail === 0) {
+			// A solo re-run only clears a failure if it GENUINELY succeeded:
+			// zero failures AND a clean exit. A file that crashed before any
+			// test ran (collection error, missing import) emits no "N failed"
+			// lines — fail parses as 0 — and must NOT be adjudicated as a
+			// flake. That masking is a silent-success failure mode.
+			const soloClean = solo.fail === 0 && solo.code === 0 && !solo.timedOut;
+			if (soloClean) {
 				totalFail -= r.fail; flakes.push(r.file);
 				failed.splice(failed.indexOf(r), 1);
 				console.error(`✓ solo: ${r.file}  (${solo.pass} pass / 0 fail, ${fmt(solo.ms)})  (flake)`);
 			} else {
-				console.error(`✗ solo: ${r.file}  (${solo.pass} pass / ${solo.fail} fail)`);
+				const why = solo.timedOut ? "TIMED OUT" :
+					solo.code !== 0 ? `crashed, code ${solo.code}` : `${solo.fail} fail`;
+				console.error(`✗ solo: ${r.file}  (${solo.pass} pass / ${solo.fail} fail — ${why})`);
 			}
```

## Why exit codes are the right axis

Exit codes are contractual: pytest exits 2 on interrupted collection, 1
on test failures, 0 on pass; node's runner exits 0 on pass; a signal
kill resolves `code` to `null`, which `!== 0`. Deepening the output
regex (e.g. also matching "ERROR") repeats the bug against the next
output-format change. Genuine flakes still adjudicate cleanly: real
failures in the parallel lane, clean solo pass → flake. Timeouts and
signal kills in solo stay failed.

## Verification record

On this repo (single test file, crashed on import at the time):

| Stage | Gate output | Exit |
|---|---|---|
| baseline | crash → `✓ solo … (flake)` | 0 (masked) |
| detector fix | `✗ solo … crashed, code 2` under FAILED FILES | 1 |
| test fixed | `✓ … (1 pass / 0 fail)` | 0 |

Upstream PR: _(link when filed)_.
