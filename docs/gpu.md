# GPU (optional) — DFlash2 spec-decode with CUDA / ROCm / Vulkan / Metal

Everything in this repo was **measured on CPU-only hardware** (ucs03). The
llama.cpp flags below come from the fork we build (`unslothai/llama.cpp`,
`glm5next/upstream`) and work on any GGML GPU backend — the dflash path has
no CPU-specific code: `--ngl` offloads target layers, `--ngld` offloads
draft layers, and the drafter's `ctx_other` (borrowed target embeddings /
lm_head) spans backends like any other draft model. What is **unmeasured**
here is GPU t/s; bring your own numbers via `scripts/gpu_ab.sh` and
`scripts/ab_8prompt.py`.

## TL;DR recipes

| Setup | Flags (added to the verified recipe in `README.md`) |
|---|---|
| GPU fits the whole target | `-ngl 99` (draft auto-offloads too — `--spec-draft-ngl` defaults to `auto`) |
| Big target, small GPU (our IQ4_XS 147 GB case) | `-ngl <fits> -ngld all` — keep the bandwidth-bound target on CPU, put the 1B drafter on GPU |
| Multi-GPU / pick devices | `--device <list>` + `-ngl all`, draft on its own card with `-devd <dev>` |
| No usable GPU (our box) | nothing — `-ngl`/`-ngld` are accepted and ignored with a warning |

```bash
# target fully on GPU (e.g. 24 GB card + IQ4_XS partial, or a 192 GB card)
llama-server -m GLM-5.3-Flash-*.gguf -md dflash2-glm-f16.gguf \
  --spec-type draft-dflash -ngl 99 \
  --ctx-size 131072 --jinja --flash-attn on

# hybrid: CPU target + GPU draft — the interesting mode for this model
llama-server -m GLM-5.3-Flash-*.gguf -md dflash2-glm-f16.gguf \
  --spec-type draft-dflash -ngl 0 -ngld all \
  --threads -1 --numa distribute --load-mode mlock \
  --spec-draft-n-max 4 --spec-draft-p-min 0.4 --top-k 20
```

## Flag matrix (dflash-relevant, from `common/arg.cpp` in the fork)

| Flag | Default | Notes |
|---|---|---|
| `-ngl, --n-gpu-layers N` | 0 | target layers in VRAM; `99`/`all` = everything that fits |
| `--spec-draft-ngl, -ngld N` | **auto** | draft layers in VRAM. `auto` places what fits — with a CPU target and a free GPU, the drafter lands on the GPU by itself |
| `-devd, --spec-draft-device LIST` | none | dedicated device(s) for the draft ctx (separate from the target's `--device`) |
| `--spec-draft-type-k/v, -ctkd/-ctvd` | f16 | draft KV cache type (small: 5 layers, block of 8) |
| `--spec-draft-backend-sampling` | off | GPU-side draft sampling — **silently a no-op for DFlash2 by design** (`common/speculative.cpp:1034`: dflash2 reads its selector lattice from `h_nextn`, never raw draft logits) |
| `-cmoe / --cpu-moe`, `--spec-draft-cpu-moe` | off | pin (draft) expert weights to CPU RAM — only relevant to the DSpark-moe draft variant; the GLM DFlash2 draft is dense |
| `--fit <on\|off>` | on | auto-sizes ctx to free VRAM; set `--fit off` when you want fixed `--ctx-size` semantics with `-ngl` |
| `--list-devices` | — | device discovery; prints `(none)` cleanly on CPU-only builds (verified on our build) |

## VRAM budget (what the drafter costs)

| Component | Size |
|---|---|
| `dflash2-glm-f16.gguf` draft weights (81 tensors, 1B) | ~2.2 GB |
| draft KV ring @ 128k ctx, f16 (5 layers, GQA) | ~1–2 GB |
| target embeddings/lm_head borrowed via `ctx_other` | 0 extra — lives with the target |

So `-ngld all` needs ~4 GB of headroom. The target is the VRAM monster
(IQ4_XS ≈ 147 GB split-file); `-ngl` is where you dial that in.

## Re-tune the block config on GPU — do NOT copy ours blindly

Our locked `n_max 4 + p_min 0.4 + top-k 20` exists **because** of RC-1
(`research/07`): on a bandwidth-bound CPU, verifying an 8-token block costs
~2.75× a single token, so wide blocks lose. On a GPU the verify batch is
nearly free until you exceed compute/SM utilization, so:

- start from the upstream default block (`n_max 7`, `p_min 0` — the trained
  block_size is 8 either way) and compare against 4;
- expect higher acceptance on fp16/bf16-class targets (the published
  5.4–5.8 figures are GPU paths; our 2.76 is the IQ4_XS-logit ceiling,
  corroborated by `brandonmusic/GLM-5.3-Flash-tr3-4bpw`);
- `-ngld all` + a fully-offloaded target is the config incoai/SGLang
  publish against — closest to their numbers.

## Measuring it on your hardware

```bash
# quick A/B against a running server (any host with the API):
python3 scripts/ab_8prompt.py --port 8100

# or let the harness boot/teardown the server for you:
bash scripts/gpu_ab.sh --ngl 0 --ngld all --tag cpu-target-gpu-draft

# decode-heavy ruler mirroring the published methodology:
python3 scripts/bench_gsm8k_mirror.py --url http://127.0.0.1:8100
```

Score against the tier ladder in `benchmarks/acceptance-gate.md` and append
your row to `research/08-improvement-tracking.md` (append-only trend log).
Correctness gates that transfer unchanged: `scripts/bench_greedy_lossless.py`
(distribution-lossless + self-consistency; bitwise spec-on vs spec-off is
**not** expected on any backend — see REQ-SD-4 in `openspec/changes/
add-dflash2-support/specs/spec-decode/spec.md`).

## Known caveats

- **No CUDA/ROCm hardware here** — ucs03's only "GPU" is a Matrox G200
  server VGA. All GPU claims above are code-verified against the fork, not
  run. Treat first GPU results as data, not validation.
- `--flash-attn on` + `-ngl` > 0 on a dflash target exercises the
  cross-attention/`embd` capture path on GPU; if you hit divergence, test
  `--flash-attn off` as the discriminator and note it in `research/08`.
- `ctx_other` + multiple GPUs: the draft's borrowed tensors stay where the
  target's embeddings live. If that costs PCIe round-trips, try `-devd`
  pinning the draft to the same card as layer 0.
