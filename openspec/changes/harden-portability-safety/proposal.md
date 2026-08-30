# Proposal: harden-portability-safety — run anywhere, publish deliberately

## Why

The QA pass found the harness pinned to one machine: `/mnt/ollama/...`
absolute paths in ten files, several with no override at all — the
conversion gate scripts (`check_tensor_inventory`, `diff_gguf_meta`,
`check_conv_base`) hardcode the `gguf-py` sys.path, so the "re-verify any
rebuild" gates behind the HF publication cannot run anywhere else; the
golden-chain regeneration scripts hardcode checkpoint, config, and draft
paths; `tests/golden/build.sh` hardcodes the fork tree. The consequence:
the repo's correctness evidence ("PASS @ 1e-3, golden chain") is
regenerable only on the author's filesystem, and a fresh contributor
cannot re-verify anything without editing files first.

Two outward-facing flows had no deliberate step: `huggingface/upload.sh`
auto-flipped the repo PUBLIC as step [4/4] the moment its gates passed
(irreversible-ish, no confirmation), and the production systemd unit
binds `--host 0.0.0.0` — llama-server has no authentication, so this is
a free LAN-wide LLM endpoint sitting on a 147 GB mlock'd process.

## What Changes

- **Env-var escapes** (applied 2026-08-31): every machine-local absolute
  path is overridable — `GGUF_PY` (gguf-py sys.path, five files),
  `DFLASH2_CKPT` / `DFLASH2_CONFIG` / `DFLASH2_DRAFT_GGUF` /
  `DFLASH2_TARGET_GGUF_DIR` (artifacts), `LLAMACPP` (fork tree in
  `build.sh`), `SRC` (upload source). Defaults still point at the
  measurement box, so nothing there changes.
- **Deliberate publish** (applied 2026-08-31): step [4/4] of `upload.sh`
  requires a typed `yes` before flipping public; `FLIP_PUBLIC=1` is the
  scripted escape. Default deny; abort leaves the repo private.
- **Loopback by default** (pending): shipped systemd units bind
  `127.0.0.1`; LAN exposure becomes a commented opt-in with the
  no-auth caveat written next to it.
- **Portability docs** (pending): README table of every env var, and a
  `docs/golden-regen.md` end-to-end recipe for regenerating the golden
  chain on a non-`/mnt` machine (env vars + the two upstream artifacts;
  no file edits).

## Impact

- AF: `scripts/check_tensor_inventory.py`, `scripts/diff_gguf_meta.py`,
  `scripts/check_conv_base.py`, `tests/golden/test_hc_collapse.py`,
  `tests/golden/sglang_ref_dump.py`, `tests/golden/make_mock_target.py`,
  `tests/golden/build.sh`, `huggingface/upload.sh`, `systemd/*.service`
  (bind address), `README.md`, new `docs/golden-regen.md`.
- OUT: the k1/w3/gpu_ab harnesses (already env-overridable — unchanged),
  the llama.cpp fork itself, any change to published benchmark numbers
  (this change touches no measurements), the HF publication itself
  (still deferred until the user says go).
