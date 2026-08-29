# GLM-5.3-Flash Target-Feature Extraction Analysis for DFlash2

This report analyzes the GLM-5.3-Flash (glm5next) architecture and the associated `llama.cpp` implementation in the unsloth fork to determine the feasibility of supporting a 5-layer target-feature extraction for DFlash2.

## 1. GLM-5.3-Flash Architecture Implementation

The implementation in `/mnt/ollama/models/llama-cpp-glm5/src/models/glm5next.cpp` reveals a complex hybrid architecture.

### Layer Composition
- **Total Layers**: Typically 45 layers (verified for `LLM_TYPE_313B_A17B` in `glm5next.cpp:102`).
- **Dense Leading Layers**: The first `hparams.n_layer_dense_lead` layers are dense (`glm5next.cpp:201`).
- **Attention Types**:
    - **KDA (Linear Attention/Recurrent)**: Used in layers where `hparams.is_recr(il)` is true. Implemented in `build_kda_layer` (`glm5next.cpp:232`).
    - **DSA (DeepSeek-style Sparse Attention)**: Used in other layers. Implements MLA (Multi-head Latent Attention) and an indexer-based sparse path. Implemented in `build_dsa_layer` (`glm5next.cpp:468`).
- **MoE Structure**: 
    - **Experts**: 288 total experts (`glm5next.cpp:102`).
    - **Active Experts**: 8 active + 1 shared expert.
    - **Tensors**: Uses `ffn_gate_exps`, `ffn_up_exps`, `ffn_down_exps` for routed experts and `ffn_gate_shexp`, `ffn_up_shexp`, `ffn_down_shexp` for the shared expert.
- **mHC (Manifold-Constrained Hyperconnections)**:
    - Implemented via `hc_attn_fn`, `hc_attn_base`, `hc_attn_scale` for attention and similar tensors for FFN.
    - The state is managed as multiple streams (controlled by `dsv4_hc_mult`).
- **MLA Attention**: 
    - Uses LoRA-compressed Q and KV.
    - Tensors: `wq_a`, `wq_b`, `wkv_a_mqa`, `wk_b`, `wv_b`.
- **Indexer**:
    - A complex k-pool indexer that selects top-k pools of keys (`glm5next.cpp:331`).

## 2. `llama_set_embeddings_layer_inp` Implementation

### Mechanism
- **API**: `llama_set_embeddings_layer_inp(ctx, lid, value)` enables/disables extraction for layer `lid`.
- **Hook Point**: In `glm5next.cpp`, hooks are placed at the start of the layer loop (pre-attention) and after the last layer.
- **Precision**: 
    - For $il < n\_layer$: It captures the state **before** the $il$-th layer's attention mechanism.
    - **mHC Handling**: Because `glm5next` uses multiple streams, `build_hc_mean` is called (`glm5next.cpp:613`) to collapse the manifold streams into a single mean vector of size `n_embd`.
- **Buffer**:
    - Tensors are added to the graph via `res->t_layer_inp[il]`.
    - During the output phase, `llama_context::extract_layer_inputs` (`llama-context.cpp:2203`) copies these tensors from the backend to a pre-allocated CPU buffer `embd_layer_inp`.
- **Retrieval**: The caller uses `llama_get_embeddings_layer_inp(ctx, lid)` to get a pointer to the float buffer for the current batch.

## 3. `llama_set_embeddings_nextn`

- **Function**: Enables extraction of the output of the "NextN" (MTP) layers or the final hidden state if no NextN layers are present.
- **Masked Mode**:
    - If `masked=true`: Only extracts for tokens where `batch.logits[i]` is true (standard for output tokens).
    - If `masked=false`: Extracts for all tokens in the batch.
- **Usage**: Used by DFlash2 to read its selector lattice from the hidden states of the draft model.

## 4. Arbitrary Layer Extraction Support

The `glm5next` model class supports extraction at **arbitrary layers**.
- The loop in `glm5next.cpp:611-659` checks `cparams.embeddings_layer_inp[il]` for every layer.
- There are no architectural restrictions in the code that prevent extracting from layers 5, 14, 24, 33, and 42.

## 5. Output Head and Embeddings

- **lm_head**:
    - Tensor: `model.output` (`glm5next.cpp:138`).
    - Shape: `{n_embd, n_vocab}`.
    - It is applied to the result of a final `output_norm` (RMSNorm).
- **Embedding Table**:
    - Tensor: `tok_embd` (`glm5next.cpp:135`).
    - Shape: `{n_embd, n_vocab}`.
    - Rows for specific tokens (like `mask_token_id = 154856`) can be gathered by indexing into this tensor.

## 6. Existing DFlash Implementation Analysis

### Current State (`common/speculative.cpp`)
- **Layer Constraint**: The current code has a hard-coded check: `if (target_layer_ids_n != 3) { throw ... }` (line 472 in `draft-eagle3`, and similar logic in DFlash).
- **Extraction Logic**:
    - It iterates over `target_layer_ids_n` and calls `llama_set_embeddings_layer_inp` for each.
    - It gathers features into `features_buf` with shape `[n_tokens, target_layer_ids_n * n_embd_tgt]`.
- **DFlash2 Specifics**:
    - `is_dflash2` is determined by `selector_top_k > 0` (`speculative.cpp:984`).
    - DFlash2 uses `llama_set_embeddings_nextn(ctx_dft, true, /*masked*/ !is_dflash2)` to get the selector lattice.

### Changes needed for 5-layer DFlash2:
1. **Remove Hardcoded Count**: Remove the `target_layer_ids_n != 3` assertion.
2. **Dynamic Buffer Sizing**: The `features_buf` is already resized based on `target_layer_ids_n`, so it will naturally handle 5 layers if the model metadata specifies them.
3. **Metadata**: Ensure the draft model's `target_layer_ids` metadata is updated to contain the 5 target indices.
4. **Input Embeddings**: Since `glm5next`'s extraction already handles mHC mean collapsing, no changes are needed to the model class itself; the existing `llama_set_embeddings_layer_inp` API is sufficient.

## Summary Table

| Feature | GLM-5.3-Flash Implementation | DFlash2 Compatibility |
| :--- | :--- | :--- |
| **Layer-wise Extract** | Pre-layer, mean-collapsed mHC | High |
| **Arbitrary Layers** | Fully supported via `cparams` | High |
| **Target Indices** | Managed by `llama_model_target_layer_ids` | Requires metadata update |
| **Feature Shape** | `[n_tokens, n_embd]` per layer | High |
| **Lattice Extraction** | via `llama_set_embeddings_nextn` | High |
