# Golden-chain regeneration (REQ-HP2)

Regenerates every fixture in `tests/golden/fixtures/` (gitignored by
design) and re-runs the two-arm comparison that backs the README's
"Draft correctness vs SGLang reference — PASS @ 1e-3" claim. Everything
below is driven by the [environment overrides](../README.md#environment-overrides)
— no file edits. Upstream artifacts you must supply yourself:

| Artifact | Source | Used by |
|---|---|---|
| DFlash2 checkpoint | `incoai/GLM-5.3-Flash-DFlash2` (BF16 safetensors) | reference arm |
| Draft GGUF | your own conversion via llama.cpp's `convert_hf_to_gguf.py` (`DFlash2DraftModel`), verified by the gate scripts | mock target, replay arm |
| Target GGUF (147 GB) | `unsloth/GLM-5.3-Flash-GGUF` (UD-IQ4_XS) | hiddens fixture only |
| llama.cpp fork build | `unslothai/llama.cpp` `glm5next/upstream` lineage | harness build |
| Python | numpy + pytest everywhere; **torch** for the reference arm (the box used its sglang venv) | both arms |

Solo-run rule applies throughout: the 147 GB load (step 2) must not
overlap any running server.

## Step 1 — build the C++ harnesses

```bash
LLAMACPP=/path/to/llama-cpp-fork tests/golden/build.sh
# → tests/golden/build/{dump_target_hiddens,replay_dflash2}
```

## Step 2 — hiddens fixture (needs the target GGUF; run solo)

Deployment-realistic hiddens: same quantization, same extraction path
(`llama_get_embeddings_layer_inp`, post `build_hc_mean`) the draft sees
in production. Fixed canned agentic prompt; override with
`GOLD_PROMPT_FILE`.

```bash
GOLD_OUT=tests/golden/fixtures/hiddens.bin \
  tests/golden/build/dump_target_hiddens -m /path/to/GLM-5.3-Flash-UD-IQ4_XS-00001-of-00005.gguf
python3 tests/golden/save_npz.py tests/golden/fixtures/hiddens.bin tests/golden/fixtures/hiddens.npz
```

## Step 3 — reference arm (pure-torch SGLang reimplementation)

```bash
DFLASH2_CKPT=/path/to/model.safetensors \
DFLASH2_CONFIG=/path/to/config.json \
  python3 tests/golden/sglang_ref_dump.py
# → fixtures/sglang_golden.npz + fixtures/lm_head.npy
```

`lm_head.npy` is the SHARED embed/lm_head fixture, created here with
seed 123 if absent — both arms must load the same matrix, so do not
delete it between steps.

## Step 4 — mock target GGUF

Carries the shared fixture matrix + the draft's tokenizer (copied
verbatim from the draft GGUF — identical vocab ids).

```bash
GGUF_PY=/path/to/llama-cpp-fork/gguf-py \
DFLASH2_DRAFT_GGUF=/path/to/dflash2-glm-f16.gguf \
  python3 tests/golden/make_mock_target.py
# → fixtures/mock_target.gguf
```

## Step 5 — replay arm (llama.cpp production code path)

Draft-only: no 147 GB target load. Feeds the SAME fixture hiddens
through project_target_hidden → KV materialization → noise block decode
→ selector lattice walk.

```bash
GOLD_FIXTURE=tests/golden/fixtures/hiddens.bin \
GOLD_MOCK_TARGET=tests/golden/fixtures/mock_target.gguf \
GOLD_OUT=tests/golden/fixtures/llamacpp_replay.bin \
  tests/golden/build/replay_dflash2 -md /path/to/dflash2-glm-f16.gguf
```

## Step 6 — verdict

```bash
python3 tests/golden/compare_golden.py
# GATE 1e-3 on ctx_hidden: PASS   (cosine 1.0, REQ-SD-2)
# candidate overlap ≥ 15/16: PASS
```

Gate (REQ-SD-2): ctx_hidden rel/cosine ≤ 1e-3; candidate sets ≥ 15/16
overlap (tail tie-flips allowed). The proposed-token listing is
informational — divergence there is quantified per stage, not gated.

## Determinism

Fixed seeds everywhere (fixture generator 123, mock weights 123,
bench seeds 42); the reference arm is pure torch, the replay arm is the
fork's own kernels — sub-1e-3 cosine agreement is the contract, not
bitwise equality. A rebuilt fork (different compiler/ISA path) may shift
the last digits; that is within REQ-SD-2's tolerance and must NOT be
"fixed" by loosening the gate without a written re-baseline here.

## hc capture semantics (separate, cheap)

`python3 tests/golden/test_hc_collapse.py` runs without any fixtures
(synthetic weights); point `DFLASH2_TARGET_GGUF_DIR` at the target
shards to also exercise the real-weights arm (needs `GGUF_PY`).
