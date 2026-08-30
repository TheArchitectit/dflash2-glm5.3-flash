# Design: dynamic release workflow (solo vibe-coding, autonomous)

## Shape: self-paced loop, NOT a fan-out workflow

The whole box is a **single serial resource**: one 147 GB model fits in RAM at
a time (:8086 Qwen production is off-limits but even so, one big model per
window). Parallel agents would contend for RAM, ports, and the CPU, and
spec-decode benchmarks are thermally serial-sensitive. So the "dynamic
workflow" is an **event-driven state machine with one worker (the loop)**:

```
                 ┌─────────────────────────────────────┐
                 │  LOOP TICK (woken by monitor OR     │
                 │  ScheduleWakeup heartbeat)          │
                 └──────────────┬──────────────────────┘
                                │
        ┌───────────────────────┼────────────────────────┐
        ▼                       ▼                        ▼
  read queue head        check live state          advance ONE step
  (tasks.md)             (log, ports, units)       then arm wake + sleep
```

## Wake-signal contract

1. **Primary = persistent Monitor** on the log/condition that ends the current
   long-running step. Pattern:
   `until grep -qE "<DONE-TOKEN>|ERROR|Traceback" /tmp/x.log; do sleep N; done; tail -5`.
   Fires the loop the instant the step finishes — no polling the model.
2. **Secondary = ScheduleWakeup heartbeat** (fallback), delay per step cost:
   - 1500–1800s (25–30 min): long benches, model loads, HF uploads.
   - 600s: short waits (unit swaps, converter runs).
   Always pass the SAME `/loop ...` prompt so the next fire re-enters here.
   Cache guidance: past 300s the context is re-read uncached, so we lean on
   the Monitor and treat the heartbeat purely as a hang catcher, not a poll.
3. **Event messages are NOT user input.** A `<task-notification>` wakes the
   loop to handle the step, then re-arm; it never counts as approval to
   publish or to touch :8086.

## Step protocol (every tick does exactly this)

1. Read queue head + verify live state (never trust the last tick's memory —
   a `set -e` chain can die silently between ticks; check log file EXISTS,
   units ACTIVE, port BOUND).
2. **Solo-run discipline** (hard, inherited from dev gate): before any server
   start — :8086 not active, target port free, `free -g` ≥ 200 GB avail.
   One big-model server at a time. Chain scripts VERIFY each swap step
   (poll `is-active`, health-gate) instead of trusting `systemctl` exit code —
   the "Job canceled" teardown race proved this necessary.
3. Do the step. Emit a DONE-TOKEN into a log the Monitor greps.
4. On completion: write the result to `research/08-improvement-tracking.md`
   (append-only), `git commit` (one commit per step, Co-Authored-By line).
5. **Restore production :8100** if the step used a swap window.
6. Re-arm (Monitor if a new long step started; heartbeat always) OR stop the
   loop if the queue is empty / an escalation fired.

## Escalation triggers — STOP the loop, surface to the user, do not proceed

**Pre-authorized by standing user instruction** ("once we are done with v0.0.1,
lets make sure we use a secret scanner, and then make a public repo"): the
close-out chain — scan → create public repo → push → HF upload → tag — runs
autonomously WITHOUT re-asking, gated only on: v0.0.1 measurement complete,
DevGate green, scan demonstrably clean. It REPORTS links on completion.

These still STOP and surface to the user:

- Any secret scanner finding (any tier, not just high).
- A measured step that CONTRADICTS a published number (e.g. re-measured
  acceptance diverges >20% from the table we're about to print).
- Solo-rule conflict (something is holding a port/RAM we need).
- Anything OUTSIDE the queue tail (e.g. touching :8086, new fork patches).

The loop may autonomously: run benches, convert quants, edit docs/tracker,
commit, install/start/stop the :8100/:8101 units (teardown+restore always),
run read-only secret scans, and execute the pre-authorized close-out. That's
the "adapted for solo vibe coding, run without input" latitude — bounded by
this list.

## DevGate + file-size rules (unchanged)

- `guardrails-scan`, `regression_check`, `run-tests` green before any publish
  commit (close-out chain step 2).
- Files < 300 lines soft / < 500 hard. Long scripts split by concern.

## Tiered gates (from benchmarks/acceptance-gate.md)

Acceptance/throughput are TIERED, never a single hard-fail. The only HARD
performance gate is T0 (spec must beat spec-off — no net loss). Correctness
(golden 1e-3, self-determinism) stays a hard gate. Report the honest tier;
do not fabricate a pass to unblock publish.

## Why "dynamic" and not fixed-interval

The steps are minutes-to-hours with unpredictable tails (model load depends
on page-cache state; bench length depends on generated tokens). A fixed cron
interval either polls too fast (burns context, no signal) or too slow (idles
past completion). Event-driven wake (Monitor) + a slow safety heartbeat is the
correct fit, which is exactly what `ScheduleWakeup` in the /loop skill provides.
