# Design: harden-portability-safety

## Shape: escape hatches, not relocation

Two options existed: (a) move every path into a config file, or (b) keep
author-box defaults and add env-var overrides. Chosen: (b).

## D1 — Env vars over a config file

- The harness scripts are one-shot tools invoked from shells and unit
  files; env vars compose with what the repo already does (`${BIN:-}`,
  `${MODEL:-}`, `${DRAFT:-}` in k1/w3/gpu_ab) instead of introducing a
  second configuration mechanism alongside those.
- A config file would grow a schema, precedence rules, and its own
  "where is the config" bootstrapping problem. Env vars need none.
- Defaults stay pointed at the measurement box: zero behavior change
  there (the solo-run host rule means the box must keep working
  unchanged), while any other machine gets an explicit, documented
  override surface.

Naming is uniform and boring: `GGUF_PY` (a gguf-py checkout to put on
sys.path), the `DFLASH2_*` family for model artifacts (`_CKPT`,
`_CONFIG`, `_DRAFT_GGUF`, `_TARGET_GGUF_DIR`), and reuse of the
pre-existing `LLAMACPP` / `SRC` / `BIN` / `MODEL` / `DRAFT` names rather
than inventing synonyms.

## D2 — Default-deny on the public flip

Making an HF repo public is outward-facing and effectively
un-retractable (caches, mirrors, indexes). A gate-passing script
auto-flipping visibility conflates "our checks passed" with "a human
decided to publish" — the repo's own release discipline language
("before any push", "solo-run rule") already separates those. The
confirmation is a typed `yes` (not y/enter — no muscle-memory slips);
`FLIP_PUBLIC=1` covers scripted reruns. Abort exits 1 with the repo
still private, and the message says exactly which env var skips the
prompt next time.

## D3 — Loopback default for shipped units

llama-server has no authentication primitive. `0.0.0.0` on a LAN means
anyone reachable can drive a 147 GB mlock'd process (RAM pressure on a
mlock'd model is itself a availability risk). The unit is a shipped
artifact others copy, so it should fail closed: `--host 127.0.0.1`, with
a commented `0.0.0.0` line that documents what you are accepting (no
auth; firewall it yourself). The measurement box's own exposure choice
is its operator's to re-make deliberately.

## D4 — Golden-chain regeneration becomes a documented recipe

The chain's evidence value depends on being re-runnable by someone other
than the author. `docs/golden-regen.md` states the full ladder with the
env vars only: make_mock_target (needs `GGUF_PY` + `DFLASH2_DRAFT_GGUF`),
sglang_ref_dump (needs its venv + `DFLASH2_CKPT`/`_CONFIG`), build.sh
(`LLAMACPP`), replay + compare (fixtures local). Anything the recipe
cannot do without a file edit is a bug in this change, not a docs gap.

## D5 — Relation to harden-quality-gates

The two changes touch the same files in places (test_hc_collapse.py gets
its `GGUF_PY` from this change, its lazy import from the other). The
split is by consequence, not by file: quality-gates owns "the battery
reports truth"; this change owns "the harness runs elsewhere and its
outward actions are deliberate". Tasks that landed 2026-08-31 are
recorded in both changes with cross-references, matching how the repo
backfills tasks.md for completed work.
