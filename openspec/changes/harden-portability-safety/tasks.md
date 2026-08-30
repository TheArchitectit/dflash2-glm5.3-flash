# Tasks: harden-portability-safety

Protocol: every env var shipped here is demonstrated with one override
smoke (value pointing at a wrong/missing location, correct behavior
observed) before its task is checked.

## P1 — env-var escapes (applied 2026-08-31)

- [x] `GGUF_PY` for the gguf-py sys.path in `check_tensor_inventory.py`,
  `diff_gguf_meta.py`, `check_conv_base.py`, `make_mock_target.py`,
  `test_hc_collapse.py` (lazy import + skip semantics belong to
  harden-quality-gates G1).
- [x] `DFLASH2_CKPT` default for `--ckpt` in `check_tensor_inventory.py`
  and `check_conv_base.py`; `DFLASH2_CKPT`/`DFLASH2_CONFIG` for
  `sglang_ref_dump.py`; `DFLASH2_DRAFT_GGUF` in `make_mock_target.py`;
  `DFLASH2_TARGET_GGUF_DIR` in `test_hc_collapse.py`.
- [x] `LLAMACPP="${LLAMACPP:-…}"` in `tests/golden/build.sh`.
- [x] Override smoke: `GGUF_PY=/nonexistent python3
  tests/golden/test_hc_collapse.py` → synthetic arm PASS, real-weights
  skip, exit 0. Remaining vars documented as pending their box-side
  smoke in P4.

## P2 — deliberate publish (applied 2026-08-31)

- [x] `upload.sh`: `SRC="${SRC:-…}"`; step [4/4] requires typed `yes`
  (or `FLIP_PUBLIC=1`); abort leaves the repo private and names the
  env var. Not executed against HF (publication still deferred) —
  syntax-checked only (`bash -n`); live confirmation happens at T6.

## P3 — loopback by default (applied 2026-08-31)

- [x] `systemd/llama-server-glm5-dflash2.service` and
  `llama-server-glm5-nospec.service`: `--host 127.0.0.1` with the
  `0.0.0.0` LAN opt-in as an in-unit comment carrying the no-auth caveat.
- [x] `llama-server-glm5-mtp.service` (defunct probe, still shipped) and
  the GPU template unit (the one strangers copy): same treatment.
- [x] README serving recipe notes the loopback default and points at the
  in-unit opt-in.

## P4 — portability docs (applied 2026-08-31)

- [x] README "Environment overrides" table (variable → used by → what
  it points at) + pointer to the regen doc.
- [x] `docs/golden-regen.md`: end-to-end golden-chain regeneration
  recipe using only env vars + the upstream artifacts; states which
  steps need torch vs the llama.cpp fork build; determinism/re-baseline
  notes.
- [ ] Box-side override smokes for `DFLASH2_CKPT`, `DFLASH2_DRAFT_GGUF`,
  `LLAMACPP`, `SRC` (each var set to its real alternate location once,
  script reaches the asset it names). `GGUF_PY` override already
  demonstrated off-box (P1).

## P5 — closure

- [ ] Both changes' pending tasks done or consciously re-queued;
  `openspec validate --all` green (shared with harden-quality-gates G4).
