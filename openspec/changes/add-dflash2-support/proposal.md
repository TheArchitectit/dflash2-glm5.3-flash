# Proposal: add-dflash2-support

## Why

GLM-5.3-Flash IQ4_XS (147 GB) on the CPU-only ucs03 box (dual-Xeon, ~120 GB/s) decodes at
1.32 t/s. The box is bandwidth-bound: every generated token costs one full pass over target
weights, so decode speed is fixed by memory bandwidth, not compute.

Block-diffusion speculative decoding is the direct lever on that bound: verifying N drafted
tokens costs the same single weight read as generating 1 token. DFlash2 drafts 7 tokens per
step (block_size=8) and publishes 5.78 accepted tokens/step for GLM-5.3-Flash — a projected
~7 t/s (~5.3x) with zero accuracy loss (lossless rejection sampling). The draft model is tiny
(~2.2 GB BF16, 81 tensors, 1B params) against a 147 GB target, so draft cost is negligible.

Research (rev 2, six reports in `research/`) found the port already exists: DFlash2 support
was merged into the llama.cpp lineage both our forks carry (PR #27342, commit b10f9ca58,
Aug 27 2026) — the `LLM_ARCH_DFLASH` model class with grouped dynamic conv + candidate
selector (`src/models/dflash.cpp`), the `draft-dflash` spec impl
(`common/speculative.cpp`, ~line 962), and converter handling
(`conversion/qwen.py:642-775`). The work is **conversion + validation**, not porting. No GGUF
of the GLM DFlash2 draft exists; the only published one is
`incoai/Qwen3.8-27B-DFlash2-GGUF` (different target model).

## What Changes

- **Converter run**: run the glm5 fork's existing converter
  (`/mnt/ollama/models/llama-cpp-glm5/conversion/qwen.py`, `DFlashModel`, registered for
  `DFlash2DraftModel` at qwen.py:642) on `incoai/GLM-5.3-Flash-DFlash2`, producing
  `dflash2-glm.gguf`. Vocab sourced from the GLM-5.3-Flash target dir (`--target-model-dir`,
  eagle3-style `set_vocab` delegation).
- **Possible arch alias registration**: if the converter rejects the checkpoint's arch name,
  register the alias via `@ModelBase.register(...)` — a 1-line patch. Code inspection shows
  qwen.py:642 already registers `DFlash2DraftModel`, so this is contingency only.
- **Conversion parity checks**: 81-tensor inventory vs the rev-1 table; metadata diff vs
  `incoai/Qwen3.8-27B-DFlash2-GGUF` (same converter path, known-good); golden check on the
  conv-base layout ([2,2,4096] checkpoint → GGUF `[n_embd, kernel, 2]`) and the +1 target
  layer indexing ([5,14,24,33,42] → [6,15,25,34,43]).
- **Smoke test**: `llama-server` (glm5 fork) with the GLM-5.3-Flash IQ4_XS target +
  `--spec-type draft-dflash -md dflash2-glm.gguf`; gate: server starts, `draft_n > 0` in
  timings, no asserts.
- **Validation harness** in this repo (`scripts/`): golden test vs the SGLang reference
  (CPU, torch; 1e-3 rel tol on draft hiddens + proposed path), acceptance-length gate ≥ 5.0
  over ~50 agentic prompts, greedy lossless check (spec on == spec off).
- **CPU benchmark**: ucs03 solo run — effective t/s vs the 1.32 baseline, acceptance %,
  wall-clock on the standard 3-task agentic suite.
- **Publication**: GGUF on Hugging Face with CC BY-NC-ND 4.0 attribution; release notes to
  incoai and llama.cpp.

## Impact

- **Affected specs** (capability deltas under `specs/` in this change):
  - `specs/conversion/spec.md` — new capability: DFlash2 checkpoint → GGUF conversion with
    parity gates (REQ-CONV-1..4)
  - `specs/spec-decode/spec.md` — new capability: draft-dflash spec decode on the glm5 fork
    for GLM-5.3-Flash (REQ-SD-1..4)
  - `specs/benchmark/spec.md` — new capability: solo CPU benchmark + publication (REQ-BM-1..2)
- **New files**: `scripts/` validation + benchmark harness (golden replay, acceptance runner,
  benchmark driver); `dflash2-glm.gguf` artifact (published to HF).
- **Fork changes**: none expected. The 5-vs-3 target-layer question is already resolved (the
  `target_layer_ids_n != 3` assert belongs to the Eagle3 impl; the DFlash impl asserts
  `> 0` at speculative.cpp:962). Zero C++ changes expected; the only patch contingency is a
  1-line Python-side converter alias registration (already present at qwen.py:642).
- **Out of scope**: no llama.cpp C++ porting, no new converter architecture from scratch, no
  SGLang changes, no training, no changes to the production `:8086` service (benchmarks run
  solo while it is idle).
