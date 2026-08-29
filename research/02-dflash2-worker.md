# DFlash2 Speculative Decoding Technical Report

This report details the implementation of the DFlash2 speculative decoding worker/runner in SGLang, based on the analysis of `DFlashWorkerV2` and related components.

## 1. Full Draft-Verify Loop

The DFlash2 process operates in a fixed-block manner. At each decode step:

1.  **Draft Block Preparation**: The worker prepares a block of size `block_size` (e.g., 16). It creates a sequence of input IDs where the first token is the "bonus" token (the accepted token from the previous step) and the remaining `block_size - 1` tokens are initialized to a special `mask_token_id`.
2.  **Draft Forward**: The draft model runs a forward pass on this block.
    - The draft model does not have its own LM head; it borrows the target model's `lm_head` during execution.
3.  **Draft Proposal**:
    - **DFlash2 (Selector)**: Uses a `CandidateSelector` to propose a path through a lattice of candidates. It computes unary logits for the top-K candidates and then uses a codebook-based scoring mechanism to pick the most likely sequence.
    - **DFlash (Greedy)**: Performs a greedy argmax over the borrowed `lm_head` for each position in the block to propose `block_size - 1` tokens.
4.  **Target Verification**: The target model runs a forward pass on the proposed block (the bonus token + drafted tokens). This is performed using `ForwardMode.TARGET_VERIFY`, which returns logits for all positions in the block.
5.  **Acceptance & Bonus**:
    - The target logits are compared against the drafted tokens.
    - The `accept_len` is the number of consecutive draft tokens that match the target's predictions.
    - A "bonus" token is extracted from the target's prediction at the first mismatch position (or the end of the block).
6.  **KV Cache Update**: The target hidden states from the verified block are used to update the draft model's KV cache (see section 5).

## 2. Target Hidden State Capture

DFlash2 relies on "context features" captured from the target model to seed the draft model.

- **Capture Mechanism**: During the target model's prefill (`is_extend()`) or verification phases, hidden states are captured from specific layers.
- **Layer Selection**: The number and indices of layers are determined by `DFlashDraftConfig.resolve_target_layer_ids`. The draft model typically has $L_{draft}$ layers, and it captures $L_{draft}$ corresponding hidden states from the target model.
- **Processing**: These captured hidden states are passed to `DFlashDraftModel.project_target_hidden`.
- **Tensor Flow**:
    - **Input**: `target_hidden` with shape `[N, num_context_features * hidden_size]`.
    - **Projection**: A linear layer `self.fc` projects this concatenated vector back to `hidden_size`.
    - **Normalization**: The result is passed through `self.hidden_norm` (RMSNorm).
    - **Result**: A tensor of shape `[N, hidden_size]` which serves as the embedded input for the draft model's KV materialization.

## 3. Mask Block Construction

The draft input is a hybrid of the latest known token and placeholders.

- **Block Structure**: `[bonus_token, mask_token, mask_token, ..., mask_token]`
- **Mask Token**: The `mask_token_id` is resolved from the tokenizer or config (defaulting to `<|MASK|>`).
- **Input Formation**:
    - `block_ids` is initialized with `mask_token_id`.
    - The first element `block_ids[:, 0]` is set to the `bonus_token`.
- **Embeddings**: These IDs are passed through the embedding layer.
- **Position IDs**: Formed as `prefix_lens + [0, 1, ..., block_size - 1]`.

## 4. Verification and Acceptance

Verification is the process of checking the draft's "guesses" against the target's ground truth.

- **Verification Logic**:
    - For greedy decoding, a draft token at position $t+1$ is accepted if it matches the target's argmax at position $t$.
    - **Correct Length**: $\text{accept\_len} = \text{max } k \text{ s.t. } \text{draft}[1 \dots k] == \text{target}[0 \dots k-1]$.
- **Bonus Token**: The token at `target[accept_len]` (the target's prediction for the token immediately following the last accepted draft token) is appended.
- **Mismatch**: On mismatch, only the accepted prefix and the bonus token are committed. The rest of the block is discarded.
- **Sampling**: For non-greedy sampling, SGLang uses `accept_sampling` (from `dspark_accept.py`) or `compute_dflash_sampling_correct_drafts_and_bonus`, which involves checking if the target probability of the drafted token is sufficiently high.

## 5. Draft KV Cache Management

Unlike standard speculative decoding where the draft model is a smaller version of the target, DFlash2 "materializes" its KV cache from the target's hidden states.

- **Materialization**: The target's hidden states are projected and then passed through the draft model's layer-wise KV projections (`kv_proj_only`).
- **KV Update**: In `_append_target_hidden_to_draft_kv_by_loc`, the projected K and V tensors are written directly into the draft model's KV cache at the specified `cache_loc`.
- **Statefulness**: The draft KV cache carries state across steps. It is a persistent buffer that is augmented with new target hidden states after every verification step.
- **Fused Path**: SGLang implements a `FusedKVMaterializeHelper` (Triton-based) to accelerate this projection and write process across all draft layers.

## 6. Non-Causal Attention Handling

The draft model's attention is handled via `RadixAttention`.

- **Attention Type**: Can be `DECODER` (causal) or `ENCODER_ONLY` (non-causal), determined by `_get_dflash_attention_type`.
- **Draft Block Attention**: During the draft forward, the block uses standard causal masking relative to the sequence start. However, because it's a "filling" task (filling masks), the underlying architecture can support different attention patterns if configured in `layer_types` (e.g., `sliding_attention`).

## 7. Sampling Parameters

- **Temperature/Top-P**: These parameters affect both the draft proposal (if using the selector) and the verification process.
- **Selector Sampling**: The `_SelectorDraftSampler` uses `temperatures` and a `greedy_mask` to decide whether to take the argmax or sample from the candidate lattice.
- **Verification Adjustments**: `apply_dflash_verify_logits_adjustments` ensures that logit biases and penalizers are applied consistently during the target verification phase.

## 8. CUDA-only Dependencies for CPU Port

A port to `llama.cpp` (CPU) must replace several CUDA/Triton-specific components:

1.  **Triton Kernels**:
    - `_prepare_dflash_draft_block_unchecked`: Block ID and position setup.
    - `_compute_dflash_accept_bonus_triton_unchecked`: The heavy lifting of matching draft vs target and picking the bonus token.
    - `selector_walk_triton`: The path-walking logic for DFlash2.
    - `table_qk_norm_rope_`: Fused QK normalization and RoPE.
2.  **Fused KV Materialization**: The `FusedKVMaterializeHelper` and the associated Triton kernels for writing target hiddens to the KV cache.
3.  **Cuda Graph**: The `_DflashDraftSampler` and `_SelectorDraftSampler` are designed to be captured into CUDA graphs. CPU ports will need eager execution.
4.  **TP All-Gather**: The `tensor_model_parallel_all_gather` used in the `CandidateSelector` for global top-K candidates.
5.  **Quantized Head kernels**: The specific `quant_method.apply` calls for the borrowed `lm_head`.
