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
  skip, exit 0.
- [x] Off-box mechanism smoke (2026-08-31, venv with pip `gguf`): with
  `GGUF_PY=/nonexistent`, all three conversion gate scripts import
  cleanly and fail with FileNotFoundError at the ASSET path (the
  checkpoint/GGUF, overridable via `DFLASH2_*`) — the import crash is
  gone, exactly the intended contributor failure mode.

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
- [x] Override smokes with synthetic alternates at real alternate paths
  (2026-08-31, off-box): `DFLASH2_CKPT` — check_tensor_inventory read the
  alternate safetensors header (marker tensor surfaced in output);
  `DFLASH2_DRAFT_GGUF` — full make_mock_target run, the alternate draft's
  tokenizer propagated into the output mock GGUF (only the 2.5 GB lm_head
  seed fixture faked, labeled as such); `LLAMACPP` — build.sh compiled
  against the alternate tree (its `#error` proof header fired);
  `SRC` — upload.sh passed the alternate path to `hf` (stubbed, zero
  network). `GGUF_PY` per P1.
- [ ] Real-asset full runs (fork build, real draft GGUF, 147 GB target)
  remain measurement-box territory.
- [x] REQ-HP2 papercut fixed while smoking: `make_mock_target.py` +
  `sglang_ref_dump.py` now `os.makedirs` their gitignored `fixtures/`
  dir — fresh-clone regeneration no longer needs a manual mkdir (found
  live: the smoke crashed on the missing dir).

## P5 — closure

- [ ] Both changes' pending tasks done or consciously re-queued;
  `openspec validate --all` green (shared with harden-quality-gates G4).
