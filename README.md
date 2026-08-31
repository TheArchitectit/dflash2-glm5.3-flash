# dflash2-llamacpp

Working notes, benchmark harnesses, gate scripts, and a correctness test
chain for running [incoai's DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2)
block-diffusion speculative decoding for **GLM-5.3-Flash** via
[llama.cpp](https://github.com/ggml-org/llama.cpp)
(`--spec-type draft-dflash`, already upstream in the fork we build).
Everything is **measured on CPU-only hardware**; GPU serving is supported
and documented (`docs/gpu.md`) but the numbers here are CPU — the flags are
code-verified against the fork, not GPU-benchmarked by us.

**Status: v0.0.1 shipped** (GitHub release `v0.0.1-dflash2-glm`). HF weight
publication is staged and deferred — one command when wanted:
`bash huggingface/upload.sh` (draft GGUFs only; the 147 GB target is not
distributed here).

## What works today (measured, not projected)

| Result | Value | Evidence |
|---|---|---|
| Baseline decode (no spec) | 1.32 t/s | `research/07-gap-analysis.md` |
| Spec effective, 50-prompt agentic | **1.864 t/s (+41%)** | `benchmarks/raw/acceptance_3.6_50prompt.log` |
| Spec effective, GSM8K mirror | **2.161 t/s (+64%)** | `benchmarks/raw/gsm8k_mirror.json` |
| Acceptance (mixed agentic / math / toolcall) | 2.76 / 2.69 / 3.61 tok/step | server-side journal + benches |
| Draft correctness vs SGLang reference | **PASS @ 1e-3** (cos 1.0) | golden chain, `tests/golden/` |
| Losslessness | distribution-level; each arm 100% self-deterministic | REQ-SD-4 + `research/08` |

Honest caveats: the published 5.428–5.78 acceptance figures belong to other
targets/GPU paths — on this GLM target at IQ4_XS, acceptance is ~2.7 with
strong per-workload variation. Config `n_max 4 + p_min 0.4 + top-k 20` beat
full 7-token blocks on CPU because MoE verify cost scales with batch width
(`research/07` RC-1). Tier scheme + decision record: `benchmarks/acceptance-gate.md`.

## Why CPU

DFlash2 was SGLang/GPU-only; llama.cpp's dflash path runs the block-diffusion
drafter (1B, 5 GQA layers, headless — borrows the target's embeddings and
lm_head via `ctx_other`) entirely on CPU, where verifying a block costs one
target forward pass — the same weight read as generating 1 token. GPU
backends work too (and the drafter offloads independently of the target with
`-ngld all` — useful for this 147 GB model on small cards): `docs/gpu.md`.

## Repo map

| Path | What |
|---|---|
| `openspec/changes/add-dflash2-support/` | OpenSpec change: proposal, design, tasks, release plan, spec deltas |
| `research/01..08` | deep dives: model, worker, SGLang framework, target hooks, converter, ecosystem, gap analysis, improvement tracking |
| `scripts/` | conversion gate checks (`check_tensor_inventory`, `diff_gguf_meta`, `check_conv_base`) + benches (`bench_acceptance`, `bench_agentic`, `bench_gsm8k_mirror`, `bench_greedy_lossless`, `ab_8prompt`) + suite generators |
| `tests/golden/` | two-arm golden chain: pure-torch SGLang reimpl vs llama.cpp replay harness (fixtures gitignored; regenerable, fixed seeds) |
| `systemd/` | production unit (dflash2), spec-off baseline unit, defunct mtp probe (arch lacks the NextN graph — finding in `research/08`), GPU template unit |
| `benchmarks/` | tier ladder, raw JSON/log dumps, results write-up |
| `docs/gpu.md` | optional GPU serving: `-ngl`/`-ngld` flag matrix, VRAM budget, hybrid CPU-target/GPU-draft mode, why the CPU block config must be re-tuned on GPU |

## Serving recipe (verified on the box this was measured on)

```bash
llama-server \
  -m GLM-5.3-Flash-UD-IQ4_XS-00001-of-00005.gguf \
  -md dflash2-glm-f16.gguf \
  --spec-type draft-dflash --spec-draft-n-max 4 --spec-draft-p-min 0.4 \
  --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.01 --repeat-penalty 1.05 \
  --threads -1 --numa distribute --load-mode mlock --ctx-size 131072 \
  --flash-attn on --cache-type-k f16 --cache-type-v f16 --jinja
```

Full unit: `systemd/llama-server-glm5-dflash2.service` — it binds
`127.0.0.1` by default (llama-server has no auth; the in-unit comment
explains the `0.0.0.0` opt-in). Needs a llama.cpp
build with the `glm5next` arch + dflash spec paths (ours:
`unslothai/llama.cpp` `glm5next/upstream` lineage, PR #27754; DFlash2
converter support merged upstream in PR #27342).

## Environment overrides

Every machine-local path in the harness is overridable — defaults point
at the measurement box, everything else runs via these
(harness-portability REQ-HP1):

| Variable | Used by | Points at |
|---|---|---|
| `GGUF_PY` | `check_tensor_inventory`, `diff_gguf_meta`, `check_conv_base`, `make_mock_target`, `test_hc_collapse` | a llama.cpp checkout's `gguf-py/` (for `import gguf`) |
| `DFLASH2_CKPT` | `check_tensor_inventory` (`--ckpt` default), `check_conv_base`, `sglang_ref_dump` | DFlash2 `model.safetensors` |
| `DFLASH2_CONFIG` | `sglang_ref_dump` | DFlash2 `config.json` |
| `DFLASH2_DRAFT_GGUF` | `make_mock_target` | converted draft GGUF (tokenizer source) |
| `DFLASH2_TARGET_GGUF_DIR` | `test_hc_collapse` | dir with target GGUF shards (real-weights arm) |
| `LLAMACPP` | `tests/golden/build.sh` | fork tree (`include/`, `build/*.a`) |
| `SRC`, `REPO`, `FLIP_PUBLIC` | `huggingface/upload.sh` | upload source dir / HF repo / skip the public-flip prompt |
| `BIN`, `MODEL`, `DRAFT` | `k1_cost_curve.sh`, `w3_perf_attr.sh`, `gpu_ab.sh` | server binary / target GGUF / draft GGUF |

Golden-chain regeneration end-to-end: `docs/golden-regen.md`.

## Conversion

The draft GGUF is built with llama.cpp's own `convert_hf_to_gguf.py`
(`DFlashModel`/`DFlash2DraftModel` in `conversion/qwen.py`) — no separate
converter ships here. The gate scripts above re-verify any rebuild: 81-tensor
inventory, metadata parity vs `incoai/Qwen3.8-27B-DFlash2-GGUF`, conv-base
layout, `dflash.target_layers` off-by-one, `block_size=8`.

## Source material

| Item | Location |
|---|---|
| DFlash2 weights (BF16 safetensors) | `incoai/GLM-5.3-Flash-DFlash2` on HF |
| DFlash2 architecture reference | SGLang PR [#36708](https://github.com/sgl-project/sglang/pull/36708), `sglang/srt/models/dflash.py` |
| GLM-5.3-Flash GGUF (target) | `unsloth/GLM-5.3-Flash-GGUF` (UD-IQ4_XS) |
| llama.cpp `glm5next` arch | `unslothai/llama.cpp` branch `glm5next/upstream` |
| DFlash blog / paper | https://inco.ai/blog/dflash2/ |
| GPU acceptance reference (same target) | `brandonmusic/GLM-5.3-Flash-tr3-4bpw` runtime-results (5.428 GSM8K) |

## Quality gates

This repo uses the [DevGate Agentic Framework](https://github.com/TheArchitectit/DevGate-Agentic-Framework), vendored at `.devgate/` (plain copy, BSD 3-Clause — see `.devgate/LICENSE`).

**File-size limit:** all source files stay under 500 lines (soft warning at 300). When a file hits the soft limit, split it rather than squeezing toward the hard limit.

Run the gates before committing (all must pass). One-time setup on a fresh
clone: `npm install` (the semantic scan needs the devDependency `typescript`;
the gates themselves only need node + python3):

```bash
node .devgate/scripts/guardrails-scan.mjs          # pattern scan
node .devgate/scripts/semantic-scan.mjs            # TS/JS AST (no-op here: no TS/JS sources)
python3 .devgate/scripts/regression_check.py --all --pre-commit
node .devgate/scripts/run-tests.mjs
python3 scripts/check_raw_dumps.py                 # raw-dump duplication check (REQ-QG5)
```

CI (`.github/workflows/gates.yml`) runs this battery plus a gitleaks
tree+history scan on every push and PR — the gates are enforced, not
remembered.

Sprint bugs that must not regress are recorded in
`.devgate/.guardrails/failure-registry.jsonl` (append-only; `log_failure.py`).
Release discipline: secret battery over tree + full history before any push;
benchmark claims ship with their raw dumps; solo-run rule on the shared box.

## License

Code and docs in this repo: **MIT** (see `LICENSE`). Model weights are
separate artifacts with their own licenses — the DFlash2 draft derives from
`incoai/GLM-5.3-Flash-DFlash2` (**CC-BY-NC-ND-4.0**, research/eval only; no
commercial use without inco.ai), target from zai-org/GLM-5.3-Flash. The GGUF
model card carries the weight license; this repo's MIT covers only our
scripts, tests, and notes. DevGate framework: BSD 3-Clause (`.devgate/LICENSE`).
