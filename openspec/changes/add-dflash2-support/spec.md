# DFlash2 for llama.cpp — OpenSpec

## Purpose

Port incoai's DFlash2 block-diffusion draft model (`incoai/GLM-5.3-Flash-DFlash2`) to llama.cpp so GLM-5.3-Flash speculative decoding runs on CPU-only hosts.

**Why it matters on CPU:** decode is bandwidth-bound. Verifying N draft tokens costs the same single pass through target weights as generating 1 token. At DFlash2's reported 5.78 accepted tokens/step on GSM8K, GLM-5.3-Flash IQ4_XS goes from 1.32 t/s → projected ~7 t/s (5.3x).

**Why it must be us:** DFlash2 ships only as BF16 safetensors loadable by a custom SGLang branch (GPU-only). No GGUF exists. The `LLM_ARCH_DFLASH` already in llama.cpp upstream is a *DSV4-family* drafter with a different tensor vocabulary (MLA attn, MoE FFN, hyper-connections) — incoai's DFlash2 is architecturally distinct (GQA attn, dense SwiGLU FFN, two-tap grouped dynamic convolutions, candidate-lattice selector).

## Model Architecture (verified from checkpoint + SGLang reference)

Reference: `sglang/srt/models/dflash.py` (SGLang PR #36708) and checkpoint `config.json`.

### Global

| Property | Value |
|---|---|
| Architecture name | `DFlash2DraftModel` → **proposed GGUF arch: `dflash2`** |
| Hidden size | 4096 |
| Layers | 5 |
| Attention heads | 32 (head_dim 128, GQA) |
| KV heads | 8 |
| Intermediate size | 12288 (SwiGLU) |
| Vocab | 154880 (shares GLM-5.3-Flash vocab) |
| RMSNorm eps | 1e-5 |
| is_causal | **false** (bidirectional within block) |
| Sliding window | 2048, all 5 layers |
| RoPE theta | 10000 |
| Max positions | 1048576 |
| Target model | GLM-5.3-Flash (45 layers), target_layer_ids **[5, 14, 24, 33, 42]** (5 layers) |

### DFlash block config (from `dflash_config`)

| Key | Value |
|---|---|
| block_size | 8 (7 draft tokens per verification step) |
| conv_group_size | 16 (256 groups over hidden 4096) |
| conv_kernel_size (taps) | 2 |
| mask_token_id | 154856 |
| selector_rank | 256 |
| selector_top_k | 16 |
| target_layer_ids | [5, 14, 24, 33, 42] |

### Checkpoint tensors (81 total, BF16)

**Top-level:**
- `fc.weight` [4096, 20480] — projects concat of 5 target-layer hiddens (5×4096) → draft hidden
- `hidden_norm.weight` [4096] — RMSNorm after fc
- `norm.weight` [4096] — final norm
- No `embed_tokens`, no `lm_head` — draft consumes **target model embeddings** via fc

**Per layer (×5), suffix pattern `layers.{i}.`:**
- `input_layernorm.weight`, `post_attention_layernorm.weight` [4096]
- `self_attn.{q,k,v,o}_proj.weight` — [4096,4096], [1024,4096], [1024,4096], [4096,4096]
- `self_attn.{q,k}_norm.weight` [128] — per-head-dim QK norm
- `mlp.{gate,up,down}_proj.weight` [12288,4096] ×2, [4096,12288]
- `attention_conv.base_kernel` [2, 2, 4096] — [side, tap, channel], side 0=input side, 1=output side; tap 0 initialized to 1.0
- `attention_conv.kernel_projection.weight` [1024, 4096] — outputs 2·taps·num_groups = 2·2·256 = 1024
- `mlp_conv.base_kernel` [2, 2, 4096]
- `mlp_conv.kernel_projection.weight` [1024, 4096]

**Candidate selector:**
- `candidate_selector.hidden_projection.weight` [256, 4096]
- `candidate_selector.predecessor_codebook` [154880, 256]
- `candidate_selector.successor_codebook` [154880, 256]

### Forward pass (per verification step)

1. **Target feature injection.** Target model (GLM-5.3-Flash) runs its forward with `set_embeddings_layer_inp([5,14,24,33,42], true)`. At the anchor position, the *input embeddings* (pre-attention hidden states) of those 5 layers are concatenated → [1, 20480].
2. **Project + normalize.** `h0 = hidden_norm(fc(concat))` → [1, 4096]. (Note: SGLang name maps `encoder.output_norm_enc.weight` → `hidden_norm.weight`; native export aliases exist.)
3. **Mask block formation.** Input block = `[h0, mask_emb × 7]` where mask_emb is the target model's embedding row for `mask_token_id` (154856), i.e. gathered from the **target** embedding table. Positions for the 8 block slots are anchor_pos..anchor_pos+7 with RoPE.
4. **Per layer (5 layers):**
   a. Pre-norm RMSNorm (`input_layernorm`)
   b. **attention_conv.prepare**: `coef = kernel_projection(h).reshape(…, 2, taps, groups)`; side-0 dynamic conv applied to h (`_grouped_conv` with `base_kernel[0]`), side-1 coefficients stashed
   c. **GQA attention** (non-causal within block, sliding window 2048 across blocks), QK-norm before RoPE
   d. **attention_conv.finish**: side-1 dynamic conv on attn output
   e. Pre-norm RMSNorm (`post_attention_layernorm`) with fused residual
   f. **mlp_conv.prepare/finish** wrapping SwiGLU MLP identically
5. **Final norm** (`norm.weight`)
6. **Candidate lattice build** (`CandidateSelector.build_lattice`):
   - `hidden_r = hidden_projection(h)` → [8, 256] (per block slot)
   - Unary logits: draft hiddens projected through the **target lm_head** (vocab rows), NOT a draft-owned head — see `_project_candidate_logits`; top-K=16 candidates per slot via radix top-k
   - Edge scores: `score[b,e,p,c] = unary[b,e,c] + <A[pred]·hidden_r, B[c]>` where pred = candidate at previous slot (anchor id for slot 0), A = predecessor_codebook, B = successor_codebook
7. **Path sampling** (`sample_path` → `_follow_maps`): Viterbi-style walk of the K×K transition lattice picking the max-score coherent path of 7 candidate ids.

**Grouped dynamic conv (`_grouped_conv`)** — the core novelty, port exactly:
```python
blocks = h.unflatten(-1, (num_groups, group_size))          # [T, 256, 16]
coefficients = base.view(1, taps, num_groups, group_size) + delta.unsqueeze(-1)
out = coefficients[:, 0] * blocks
position = arange(T) % block_size                            # block-relative position
for tap in 1..taps-1:
    shifted = pad(blocks[:-tap], (…, tap, 0))                # shift back by tap
    out += coefficients[:, tap] * shifted * (position >= tap) # gate at block start
out = out.flatten(-2)
```
`delta` comes from `kernel_projection` output reshaped `[…, 2, taps, num_groups]`; side 0 is used in `prepare` (on the sublayer input), side 1 in `finish` (on the sublayer output).

### Acceptance / verification

Standard greedy: target model verifies the 7-token drafted path in one batched forward; accept the longest matching prefix, then sample the correction. (SGLang also supports sampled acceptance preserving distribution — phase 3 stretch goal, matching cafe-fork `--speculative-use-rejection-sampling` semantics.)

## Requirements

### REQ-1: GGUF converter (`tools/convert_dflash2_to_gguf.py`)

- Reads `model.safetensors` + `config.json` from `incoai/GLM-5.3-Flash-DFlash2`
- Emits single GGUF with:
  - `general.architecture = "dflash2"`
  - All 81 tensors mapped to llama.cpp names (see Tensor Map below)
  - Metadata: `dflash2.block_size=8`, `dflash2.conv_group_size=16`, `dflash2.conv_taps=2`, `dflash2.selector_rank=256`, `dflash2.selector_top_k=16`, `dflash2.target_layer_ids="5,14,24,33,42"`, `dflash2.mask_token_id=154856`, `dflash2.is_causal=false`, `dflash2.sliding_window=2048`
  - Vocabulary: copy GLM-5.3-Flash tokenizer (draft has none of its own)
  - Output dtype: BF16 (1.9 GB); optional Q8_0 for the conv/codebook tensors is phase-4 stretch
- **Parity gate:** converter output loads in a python reader with identical tensor shapes/dtypes and metadata keys.

### REQ-2: llama.cpp model class (`src/models/dflash2.cpp`)

- New `LLM_ARCH_DFLASH2` ("dflash2") registered in `llama-arch.cpp`, distinct from existing DSV4-family `LLM_ARCH_DFLASH`
- `llama_model_dflash2` implementing:
  - hparams: block_size, conv fields, selector fields, target_layer_ids (5 entries — assert exactly 5, error message pointing at DSV4 dflash confusion)
  - tensor load for all 81 tensors
  - build_graph implementing the forward pass above, CPU-backend-first (ggml ops: rms_norm, mul_mat, conv via elementwise ops — no custom CUDA kernels required; the grouped conv is expressible as reshape + broadcast multiply + shifted add, per the reference pseudocode)
  - non-causal attention within block: reuse existing non-causal path used by dflash arch (`llama_set_causal_attn(false)`)
  - sliding-window attention across blocks
  - QK-norm (per-head RMSNorm over head_dim 128) before RoPE
- **Parity gate:** given identical inputs (fixed seed, fp32), hidden_states after each layer match the SGLang reference within 1e-3 relative (test harness in REQ-4)

### REQ-3: spec-decode integration

Base fork: **`quimmedes/cafe-llama.cpp`** (has the multi-algorithm spec framework + `common_speculative_impl_draft_dflash` reference) **merged with unsloth's `glm5next/upstream`** (has the GLM-5.3-Flash target arch).

- New `common_speculative_impl_draft_dflash2` (fork of the existing dflash impl, ~300 lines different):
  - Reads `dflash2.target_layer_ids` metadata (5 ids) instead of asserting 3
  - Calls `llama_set_embeddings_layer_inp(ctx_tgt, id, true)` for the 5 layers on the **glm5next** target
  - Block-diffusion draft loop: inject target features → 1 forward → lattice → 7-token path
  - Verification via existing framework (`prepare_for_verify` path already handles dflash per cafe fork)
- Server flags: `--spec-type draft-dflash2 -md dflash2.gguf` (auto-detected by arch)

### REQ-4: test harness (`tests/test-dflash2.cpp` + `tests/ref/dflash2_ref.py`)

- `dflash2_ref.py`: loads checkpoint in PyTorch (CPU), runs the forward on canned inputs, dumps per-layer hiddens + final lattice path as `.bin` golden files
- `test-dflash2.cpp`: loads the GGUF, replays inputs, compares against goldens (1e-3 rel tol)
- End-to-end: acceptance-length measurement on 50 GSM8K-style prompts vs published 5.78 (GSM8K); target ≥ 5.0 (87% of published)

### REQ-5: benchmark + publish

- Benchmark: GLM-5.3-Flash IQ4_XS + DFlash2 on ucs03 (30-thread dual-Xeon): t/s vs 1.32 baseline
- If acceptance ≥ 5.0 and speedup ≥ 3x: publish GGUF to HF (CC BY-NC-ND 4.0, matching source license) + upstream PR to llama.cpp + note to incoai

## Tensor Map (safetensors → GGUF)

| safetensors | GGUF (llama.cpp names) |
|---|---|
| `fc.weight` | `fc.weight` (new tensor `LLM_TENSOR_FC`) |
| `hidden_norm.weight` | `enc_output_norm.weight` |
| `norm.weight` | `output_norm.weight` |
| `layers.{i}.input_layernorm.weight` | `blk.{i}.attn_norm.weight` |
| `layers.{i}.post_attention_layernorm.weight` | `blk.{i}.ffn_norm.weight` |
| `layers.{i}.self_attn.q_proj.weight` | `blk.{i}.attn_q.weight` |
| `layers.{i}.self_attn.k_proj.weight` | `blk.{i}.attn_k.weight` |
| `layers.{i}.self_attn.v_proj.weight` | `blk.{i}.attn_v.weight` |
| `layers.{i}.self_attn.o_proj.weight` | `blk.{i}.attn_output.weight` |
| `layers.{i}.self_attn.q_norm.weight` | `blk.{i}.attn_q_norm.weight` |
| `layers.{i}.self_attn.k_norm.weight` | `blk.{i}.attn_k_norm.weight` |
| `layers.{i}.mlp.gate_proj.weight` | `blk.{i}.ffn_gate.weight` |
| `layers.{i}.mlp.up_proj.weight` | `blk.{i}.ffn_up.weight` |
| `layers.{i}.mlp.down_proj.weight` | `blk.{i}.ffn_down.weight` |
| `layers.{i}.attention_conv.base_kernel` | `blk.{i}.attn_conv_base.weight` (shape [2,2,4096]) |
| `layers.{i}.attention_conv.kernel_projection.weight` | `blk.{i}.attn_conv_proj.weight` |
| `layers.{i}.mlp_conv.base_kernel` | `blk.{i}.ffn_conv_base.weight` |
| `layers.{i}.mlp_conv.kernel_projection.weight` | `blk.{i}.ffn_conv_proj.weight` |
| `candidate_selector.hidden_projection.weight` | `selector_hidden_proj.weight` |
| `candidate_selector.predecessor_codebook` | `selector_predecessor.weight` |
| `candidate_selector.successor_codebook` | `selector_successor.weight` |

## Risks & Unknowns

1. **Unary logits use the TARGET's lm_head.** The draft itself cannot produce candidate logits standalone — llama.cpp draft contexts currently own their vocab head. Resolution: the spec impl must run the target's lm_head on draft hiddens (the cafe framework's `ctx_dft`/`ctx_tgt` split supports cross-model calls; verify the `llama_set_embeddings_*` hooks expose what we need). *This is the #1 technical risk — prototype in Phase 2 before writing the full port.*
2. **Mask embedding comes from the target embedding table** (row 154856), not draft weights — the draft GGUF is not standalone-loadable without the target running alongside. Converter should record `dflash2.mask_token_id` and the loader must gather the mask row from target context.
3. **Feature extraction point:** `llama_set_embeddings_layer_inp` captures *input* embeddings (pre-attention) of target layers — must confirm glm5next model class supports extraction at 5 arbitrary mid-network layers, and that the SGLang training-time extraction points match (SGLang's `set_hidden_states_capture` equivalent).
4. **Non-causal attention**: existing dflash impl already sets `llama_set_causal_attn(ctx_dft, false)` — same requirement, reuse.
5. **RoPE across block slots**: positions continue from anchor (anchor_pos + slot). Confirm the reference uses continuous positions, not restarted — verify in Phase 2 golden test.
6. **License**: CC BY-NC-ND 4.0 — non-commercial, no derivatives of the *weights*. A GGUF conversion is a format conversion, not a derivative work in the quantization sense, but ND language may be read strictly; flag in the HF README and keep the converter separate from the weights.

## Phases

- **Phase 0 (done)**: repo, this spec, source analysis
- **Phase 1**: converter + golden test tensors dumped from SGLang reference (de-risks REQ-1, REQ-4 partially)
- **Phase 2**: fork merge (cafe + glm5next) — build only, no dflash2; then `llama_model_dflash2` skeleton + forward-pass parity vs goldens (de-risks #1, #3, #5)
- **Phase 3**: spec impl + end-to-end speculative decode
- **Phase 4**: benchmarks, Q8_0 conv tensors, HF publish, upstream PR

## References

- Checkpoint: `incoai/GLM-5.3-Flash-DFlash2` (HF, CC BY-NC-ND 4.0)
- SGLang reference: `sglang/srt/models/dflash.py` + `srt/speculative/dflash_worker_v2.py` (PR #36708), installed at `/mnt/ollama/models/glm-5.3-flash/sglang-venv/lib/python3.12/site-packages/sglang/`
- Target arch: `unslothai/llama.cpp` branch `glm5next/upstream` (local: `/mnt/ollama/models/llama-cpp-glm5/`)
- Spec framework: `quimmedes/cafe-llama.cpp` (local: `/mnt/ollama/models/llama-cpp-cafe/`)
- Benchmark claims: https://inco.ai/blog/dflash2/ (5.78 acceptance GSM8K, 2.79x @ c1 MATH-500)
