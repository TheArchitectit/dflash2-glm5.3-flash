# spec delta: release-workflow (ADDED)

## ADDED Requirements

### Requirement: REQ-WF-1 Serial solo-resource execution

All release-queue steps SHALL execute one at a time on the box. Before any
big-model server start: `:8086` inactive (never touched), target port free,
≥200 GB RAM available. Server chains SHALL poll unit state and health
endpoints after every start/stop rather than trusting command exit codes.

#### Scenario: swap-window chain

- **WHEN** a step needs a different server config than production
- **THEN** the chain stops production, waits for port release + memory drain,
  starts the step unit, health-gates before benching, and restores production
  on every exit path (including failure)

### Requirement: REQ-WF-2 Event-driven wake with heartbeat fallback

The dynamic workflow SHALL wake on step completion via a persistent Monitor
grepping a DONE-TOKEN (or error patterns) in the step's log, with a
ScheduleWakeup heartbeat (600–1800s, cost-scaled) armed ONLY as hang
detection. Monitor events SHALL be treated as data, never as user approval.

#### Scenario: long bench completes

- **WHEN** a background bench emits its DONE-TOKEN
- **THEN** the loop wakes within one poll interval, records results, commits,
  and either starts + arms the next step or stops with an empty queue

### Requirement: REQ-WF-3 One-step-per-tick with commit-per-step

Each tick SHALL advance exactly one queue step and end in a visible state:
appended row in `research/08-improvement-tracking.md` (or equivalent doc) +
one git commit. Ticks SHALL re-verify live state from logs/ports/units
before acting — never from the previous tick's memory.

#### Scenario: stale chain suspicion

- **WHEN** a monitored log is missing or a unit state contradicts the last
  report
- **THEN** the tick treats the step as failed/unstarted, diagnoses from ground
  truth, and relaunches defensively rather than waiting

### Requirement: REQ-WF-4 Bounded autonomy

The loop MAY, without user input: run benches and conversions, edit
docs/tracker/repo files, manage :8100/:8101 units with restore, run
read-only scans, commit. It SHALL STOP and surface to the user when:
a secret finding of any tier occurs, a re-measurement contradicts a
to-be-published number by >20%, the solo-resource contract is blocked,
or the queue is empty. Creating the public repo and the HF upload are
PRE-AUTHORIZED by the user's standing "secret scan then public repo"
instruction — the loop proceeds once the scan is demonstrably clean, and
reports (not asks) on completion.

#### Scenario: scan finds a secret

- **WHEN** any scanner reports a finding at any tier
- **THEN** the loop halts the close-out chain, shows the finding with file
  and commit, and waits — no push, no upload

#### Scenario: scan clean, push proceeds

- **WHEN** history + working-tree scans report zero findings at the release
  commit
- **THEN** the loop creates the public repo, pushes, uploads to HF, tags
  v1.0.0-dflash2-glm, and reports links

### Requirement: REQ-WF-5 Tiered gates, honest numbers

Publish-blocking gates SHALL be: golden correctness 1e-3 (hard),
self-determinism (hard), T0 no-net-loss (hard). Acceptance and throughput
SHALL publish at their measured tier with raw numbers attached; no
fabricated pass against the retired 5.0 hard-fail.

#### Scenario: acceptance below the retired threshold

- **WHEN** mean acceptance lands under 5.0 while golden correctness, self-determinism, and T0 all hold
- **THEN** the release publishes at its measured acceptance tier with the raw dump attached, and the 5.0 figure is reported as retired rather than failed
