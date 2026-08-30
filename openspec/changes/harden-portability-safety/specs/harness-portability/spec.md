# harness-portability — no machine-local path is load-bearing

## ADDED Requirements

### Requirement: REQ-HP1 Every machine-local path is env-overridable

Any absolute path in `scripts/`, `tests/`, `systemd/`, or
`huggingface/` that refers to machine-local assets (the gguf-py
checkout, checkpoints, GGUFs, the fork tree, upload sources) SHALL be
overridable via a documented environment variable, with the measurement
box's path as the default. A script SHALL NOT require file edits to run
against assets in different locations.

#### Scenario: gguf-py lives elsewhere

- **WHEN** `GGUF_PY` points at a different gguf-py checkout and a gate script (`check_tensor_inventory`, `diff_gguf_meta`, `check_conv_base`, `make_mock_target`) runs
- **THEN** the gguf package is imported from that location without editing the script

#### Scenario: fork tree relocated

- **WHEN** `tests/golden/build.sh` runs with `LLAMACPP` pointing at another llama.cpp build tree
- **THEN** the harnesses compile and link against that tree's libraries

### Requirement: REQ-HP2 Golden-chain regeneration is a documented env-only recipe

`docs/golden-regen.md` SHALL describe the full regeneration ladder
(mock target, SGLang reference dump, harness build, replay, comparison)
using only environment variables plus the two upstream artifacts
(safetensors checkpoint, draft GGUF), and SHALL state which steps need
the sglang venv and which need the llama.cpp fork build. Any step that
still requires editing a file is a defect against this requirement.

#### Scenario: regenerating off the measurement box

- **WHEN** a contributor with the upstream artifacts but no `/mnt/ollama` tree follows `docs/golden-regen.md`
- **THEN** every step runs via its documented env var and the chain completes without modifying repository files
