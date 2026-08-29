# Technical Report: Draft-DFlash Speculative Decoding Framework in Cafe-LLaMA.cpp

This report analyzes the `draft-dflash` speculative decoding implementation in the `cafe-llama.cpp` fork. This framework implements block-diffusion drafting with target feature injection.

## 1. `common_speculative_impl_draft_dflash` Lifecycle

### Construction
The implementation is defined as `struct common_speculative_impl_draft_dflash` (lines 908-1280 in `/mnt/ollama/models/llama-cpp-cafe/common/speculative.cpp`).

**Initialization sequence:**
1. **Metadata Extraction:** It extracts `target_layer_ids` and `target_layer_ids_n` from the draft model using `llama_model_target_layer_ids` and `llama_model_target_layer_ids_n` (lines 951-952).
2. **Block Size:** It reads the trained `block_size` from the `dflash.block_size` GGUF metadata key, defaulting to 16 (lines 960-965).
3. **Resource Allocation:**
   - `batch`: For noise tokens.
   - `batch_inject`: For target feature KV cache injection (line 988).
4. **Sampler Setup:** Initializes a CPU sampler chain with `top_k = 10` (lines 991-997) or offloads to the backend if `params.backend_sampling` is enabled (lines 1001-1013).
5. **Context Configuration:**
   - Enables layer input extraction on the target context: `llama_set_embeddings_layer_inp(ctx_tgt, ..., true)` for all target layers (lines 1016-1018).
   - Configures draft context: `llama_set_embeddings_nextn(ctx_dft, true, true)` and disables causal attention `llama_set_causal_attn(ctx_dft, false)` (lines 1020-1021).

### `process()` Method (KV Cache Injection)
The `process()` method (lines 1059-1161) handles the "injection" of target model features into the draft model's KV cache during prefill or verification.

1. **Feature Gathering:** For each token in the incoming batch, it extracts features from the specified `target_layer_ids` using `llama_get_embeddings_layer_inp` (line 1109). These are concatenated into `features_buf` (size `n_chunk * n_embd_enc`).
2. **Encoder Pass:** It runs `llama_encode(ctx_dft, enc_batch)` on the draft model's encoder part (line 1131).
3. **Injection:** It retrieves the encoder output via `llama_get_embeddings_nextn(ctx_dft)` (line 1138) and uses it as embeddings in `batch_inject` to call `llama_decode(ctx_dft, batch_inject)` (lines 1142-1151). This populates the draft decoder's KV cache.

### `draft()` Method (Token Generation)
The `draft()` method (lines 1163-1274) performs the actual speculative generation.

1. **Noise Batch Construction:** It builds a batch containing the last token (`dp.id_last`) followed by `block_size - 1` mask tokens (`mask_token_id`) (lines 1186-1190).
2. **Decoding:** Runs `llama_decode(ctx_dft, batch)` (line 1198).
3. **Sampling:**
   - For each position in the block, it samples tokens using the internal sampler (lines 1229, 1248).
   - It checks the confidence against `params.p_min` (lines 1225, 1260).
   - Tokens are accepted greedily (`top_k=1`) and pushed to the result vector (lines 1239-1243, 1258-1266).

## 2. Target Feature Injection

Feature injection allows the draft model to be seeded by the target model's internal representations.

- **Extraction:** The target context is told to save the input embeddings of specific layers via `llama_set_embeddings_layer_inp`.
- **Injection API Contract:**
  - `llama_get_embeddings_layer_inp(ctx, layer_id)` returns the pointer to the extracted features for the current batch.
  - The `common_speculative_impl_draft_dflash` gathers these into a buffer of size `n_tokens * (n_extract_layers * n_embd_tgt)`.
  - This buffer is passed to `llama_encode` on the draft context.
- **Tensors:** The draft model's encoder (defined in `llama_model_dflash::graph<true>`) processes these features through a projection (`fc`) and normalization (`output_norm_enc`) to produce a vector of size `n_embd_dec` per token.

## 3. Draft Input Embeddings

The draft model's embeddings are handled specifically to support the fusion of target features:

- **Masked Embeddings:** `llama_set_embeddings_nextn(ctx_dft, true, true)` is called. The `masked=true` argument indicates that the output of the embedding layer should be extracted for use in subsequent steps (like the injection process).
- **Target Embedding usage:** When `llama_decode` is called on `batch_inject`, the `embd` field of the `llama_batch` is used, bypassing the standard token embedding lookup and directly injecting the encoder's results into the decoder's processing stream.

## 4. Sampling and Verification

### Sampling
- **Top-K:** The draft sampler is initialized with `top_k = 10` (line 994).
- **Candidates:** `common_sampler_get_candidates` is used to inspect the probabilities.
- **Confidence:** A `p_min` threshold is applied. If the probability of the top candidate is below `p_min`, drafting for that sequence stops (line 1260).

### Verification (Cafe Framework)
Verification is handled by the base `common_speculative` machinery:
- **Accept/Reject:** The target model decodes the drafted tokens in a single batch. Tokens are accepted as long as the target model's sampled token matches the draft token.
- **Metrics:**
  - `n_gen_tokens`: Total tokens generated by the speculator.
  - `n_acc_tokens`: Total tokens accepted by the target.
  - `n_acc_tokens_per_pos`: tracked per position to analyze speculator efficiency.

## 5. `LLM_ARCH_DFLASH` Model Class

The `llama_model_dflash` class (defined in `/mnt/ollama/models/llama-cpp-cafe/src/models/dflash.cpp`) differs from standard transformers in several ways:

### Tensor Constants
Key tensors include:
- `LLM_TENSOR_FC`: Encoder projection layer.
- `LLM_TENSOR_ENC_OUTPUT_NORM`: Encoder normalization.
- `LLM_TENSOR_DSPARK_MARKOV_W1/W2`: Markov head weights for DSpark (lines 106-107).
- `LLM_TENSOR_DSPARK_CONF_PROJ`: Confidence head projection (line 109).

### Forward Pass Differences
- **Non-Causal Attention:** The decoder uses non-causal attention (`llama_set_causal_attn(ctx_dft, false)`). This is critical for block-diffusion where tokens in a block can attend to each other.
- **Dual Mode Graph:** The model uses different graphs based on the batch type:
  - `LLM_GRAPH_TYPE_ENCODER`: Implements the feature fusion (Projection $\to$ Norm).
  - `LLM_GRAPH_TYPE_DECODER`: Handles both KV injection (if `ubatch.embd` is present) and noise-block diffusion.
- **KV Injection Path:** When embeddings are provided, it skips the standard transformer block and directly computes $K = W_k \cdot \text{embd}$ and $V = W_v \cdot \text{embd}$, injecting them into the cache (lines 385-431).

## 6. Wiring and Parameters

- **Command Line Flags:** `--spec-type draft-dflash` or `draft-dspark` triggers the use of these implementations.
- **Contexts:** Requires both a target context (`ctx_tgt`) and a draft context (`ctx_dft`).
- **Draft Context Construction:** The draft context is initialized from a separate DFlash GGUF model.

## 7. Interface for New Draft Implementations

To implement a new speculator (e.g., `common_speculative_impl_draft_dflash2`), the following virtual methods from `common_speculative_impl` must be implemented:

| Method | Purpose | Requirement for DFlash2 |
| :--- | :--- | :--- |
| `begin()` | Initialize per-sequence state | Handle prompt alignment. |
| `process()` | Update draft state from target | Implement feature extraction $\to$ encoder $\to$ KV injection. |
| `draft()` | Generate candidate tokens | Implement noise batch $\to$ decode $\to$ sample. |
| `accept()` | Update state after verification | No-op for DFlash usually, but useful for state recovery. |

**Checklist for `dflash2` integration:**
- [ ] Define new `common_speculative_type` constant.
- [ ] Implement `common_speculative_impl` subclass.
- [ ] Handle GGUF metadata (block size, layer IDs).
- [ ] Implement target feature extraction and injection in `process()`.
- [ ] Implement block-based generation in `draft()`.
- [ ] Register the implementation in `common_speculative_init` (lines 2461-2580).