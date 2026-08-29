# DFlash2 Model Technical Report

This report provides the exact implementation details of the DFlash2 draft model based on the SGLang reference implementation. This is intended as the ground truth for a C++ (llama.cpp) port.

## 1. DFlashAttention.forward()

### Computation Order
The attention mechanism follows a precise sequence of operations.

1.  **QKV Projection**: 
    Input `hidden_states` are projected using `qkv_proj` (a `QKVParallelLinear` layer). This results in a single fused tensor containing Query, Key, and Value.
    - Line 283: `qkv, _ = self.qkv_proj(hidden_states)`

2.  **Normalization and RoPE (Divergent Paths)**:
    The implementation handles this differently based on the hardware and data types:

    - **NPU Path (`forward_prepare_npu`)**:
      Uses a fused kernel `split_qkv_rmsnorm_rope`.
      - Line 262: `q, k, v = split_qkv_rmsnorm_rope(...)`
      - This kernel performs: **Split $\to$ RMSNorm (Q and K) $\to$ RoPE**.

    - **Optimized CPU/GPU Path (`table_qk_norm_rope_`)**:
      Used when `use_table_qk_norm_rope` is True and dtype is `bfloat16`.
      - Line 289: calls `table_qk_norm_rope_`
      - **Exactly as implemented in the Triton kernel (`_table_qk_norm_rope_kernel`)**:
        - Line 1113: `x = tl.load(row + d_ar).to(tl.float32)` (Load)
        - Line 1113: `ms = tl.sum(x * x, 0) / D` (Mean Square)
        - Line 1114: `inv = 1.0 / tl.sqrt(ms + EPS)` (Inverse RMS)
        - Line 1117-1118: `x1 = ... * inv * w1`, `x2 = ... * inv * w2` (**RMSNorm applied before RoPE**, multiplied by the learned weight).
        - Line 1121-1122: `o1 = x1 * cos - x2 * sin`, `o2 = x2 * cos + x1 * sin` (**RoPE applied after RMSNorm**).

    - **Fallback Path**:
      - Line 302: `q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)`
      - Line 303: `q, k = apply_qk_norm(q, k, self.q_norm, self.k_norm, self.head_dim)`
      - Line 304: `q, k = self.rotary_emb(positions, q, k)`
      - Order is explicit: **Split $\to$ RMSNorm $\to$ RoPE**.

3.  **Attention Computation**:
    Uses `RadixAttention`. 
    - Scaling factor $\text{scaling} = \text{head\_dim}^{-0.5}$ (Line 218).
    - For specific models (Nemotron 3.5), an `attention_sink_bias` may be passed to the attention kernel (Lines 308-310).
    - Sliding window semantics: if `sliding_window_size` is set in the config, it is passed to `RadixAttention` (Line 253).

4.  **Output Projection**:
    - Line 312: `output, _ = self.o_proj(attn_output)`

### Summary of QK Flow: 
`Projection` $\to$ `RMSNorm` $\to$ `RoPE` $\to$ `Attention` $\to$ `Output Projection`.

---

## 2. DFlashAttention KV Handling

- **KV Cache**: SGLang uses `RadixAttention`, which manages the KV cache internally.
- **Positions**: `positions` are passed as a tensor and used by the rotary embedding layer to fetch the correct `cos`/`sin` values.
- **`kv_proj_only`**: (Lines 320-344) For materializing context tokens into the draft KV cache.
  - If weights are unquantized, it slices the `qkv_proj` weight to only compute K and V, avoiding the Q computation.
  - Slice indices: `self.q_size` to `self.q_size + 2 * self.kv_size`.

---

## 3. DFlashGroupedConv

This is a dynamic depthwise K-tap convolution applied to blocks.

### Lifecycle
1.  **`prepare(hidden_states)`**: (Lines 456-464)
    - Project input: `coefficients = self.kernel_projection(hidden_states).reshape(..., 2, taps, num_groups)`
    - Convolve input: returns `_convolve(hidden_states, coefficients[..., 0, :, :], side=0)` and the coefficients for the finish step.
2.  **`finish(hidden_states, coefficients)`**: (Lines 465-467)
    - Convolve output: returns `_convolve(hidden_states, coefficients, side=1)`.

### Internal `_grouped_conv` Implementation (Lines 396-408)
- **Kernel Weights**: `base_kernel[side]` is a learned parameter of shape `[taps, hidden_size]`.
- **Dynamic Component**: `delta` is the output of `kernel_projection`.
- **Final Weight**: $\text{coefficients} = \text{base\_kernel} + \text{delta}$ (Line 398).
- **Reshape**: `hidden_states` are unflattened to `[batch, num_groups, group_size]`.
- **Convolution Algorithm**:
  - Base case ($\text{tap}=0$): $\text{out} = \text{coefficients}[:, 0] \times \text{blocks}$.
  - Recursive taps: For $\text{tap} \in [1, \text{taps})$:
    - $\text{shifted} = \text{Pad}(\text{blocks}[:-tap], \text{left=tap})$.
    - $\text{out} = \text{out} + \text{coefficients}[:, tap] \times \text{shifted} \times (\text{position} \ge tap)$.
- **Gating**: The condition `(position >= tap)` ensures the convolution does not wrap or look back across the block boundary.

---

## 4. DFlashDecoderLayer.forward

The layer implements a Pre-Norm architecture with fused residuals.

**Exact Residual/Norm Order (Lines 502-542):**
1.  **Input Norm**: `hidden_states = self.input_layernorm(hidden_states, residual)` (Residual is added here).
2.  **Attention Conv Prepare**: If `attention_conv` exists: `hidden_states, attention_kernel = self.attention_conv.prepare(hidden_states)`.
3.  **Attention**: `attn_out = self.self_attn(...)`.
4.  **Attention Conv Finish**: If `attention_kernel` exists: `attn_out = self.attention_conv.finish(attn_out, attention_kernel)`.
5.  **Post-Attention Norm**: `hidden_states, residual = self.post_attention_layernorm(attn_out, residual)`.
6.  **MLP Conv Prepare**: If `mlp_conv` exists: `hidden_states, mlp_kernel = self.mlp_conv.prepare(hidden_states)`.
7.  **MLP**: `hidden_states = self.mlp(hidden_states)`.
8.  **MLP Conv Finish**: If `mlp_kernel` exists: `hidden_states = self.mlp_conv.finish(hidden_states, mlp_kernel)`.

---

## 5. DFlashDraftModel.forward

**Input Embedding Path (Lines 683-720):**
- If `input_embeds` is provided, it is used directly.
- Otherwise:
  - If `self.embed_tokens` (VocabParallelEmbedding) is present, it is used.
  - Otherwise, it raises a `ValueError` stating that `input_embeds` (target embeddings) are required.

**Positions and Masking**:
`positions` are passed through the layers. Mask token handling is typically managed upstream in the speculative worker.

---

## 6. CandidateSelector (DFlash2)

The selector computes transitions between proposals.

### `build_lattice` (Lines 976-1003)
Computes the edge score between candidates.
**Exact Math (from `_score_edges` Lines 910-931):**
$\text{score}[b, e, p, c] = \text{unary\_logits}[b, e, c] + \langle \text{predecessor\_codebook}[\text{pred}] \times \text{projected\_hidden}, \text{successor\_codebook}[c] \rangle$
- $\text{unary\_logits}$: Base probabilities for the candidate token.
- $\text{pred}$: The candidate chosen for the previous slot (or the verified anchor for slot 0).
- $\text{projected\_hidden}$: `hidden_states` projected to `state_rank` via `hidden_projection`.
- $\text{successor\_codebook}[c]$: The codebook vector for candidate $c$.

### `sample_path` (Lines 1004-1064)
Walks the lattice to extract tokens.
**Sampling Logic (`_selector_walk_kernel` Lines 249-290):**
1.  **Slot 0**:
    - Selects `index` based on `scores[:, 0, 0]` (Greedy $\to$ argmax, Sample $\to$ softmax).
2.  **Slot $n > 0$**:
    - The `previous` index is used to look up the scores for the current slot.
    - $\text{scores} = \text{scores\_ptr}[(\text{row} \times \text{slots} + \text{slot}) \times \text{top\_k} + \text{previous}]$.
    - A new `index` is sampled from these scores.
3.  **Candidate Extraction**:
    - $\text{token} = \text{candidate\_ids}[\text{row}, \text{slot}, \text{index}]$.

**Dimensions**: `candidate_ids` [b, slots, top\_k]. The final 7 draft tokens are simply the result of the walk across the slots.

---

## 7. Logits Projection and Vocab Restriction

The draft model does not have its own LM head.
- `_project_candidate_logits` (Lines 88-102):
  - Projects the draft hidden state using the **TARGET's** `lm_head`.
  - **Restriction**: It restricts the output to the original vocabulary size `num_org` (slicing the weight for dense heads or masking for quantized heads).

---

## 8. Target Feature Integration

**`project_target_hidden` (Lines 663-681):**
1.  Input `target_hidden`: Concatenated hidden states from $K$ target layers.
2.  Projection: `projected = self.fc(target_hidden)`.
3.  Normalization: `return self.hidden_norm(projected)`.

**`prepare_context_hidden_for_kv` (Line 658):**
In `DFlashDraftModel`, this is a no-op. In `DFlashLagunaForCausalLM` (Line 876), it applies `layer.input_layernorm(ctx_hidden)`.

---

## 9. DFlashLaguna Architecture

`DFlashLagunaAttention` (Lines 804-854) adds a learned gating mechanism.
- **Gating Logic**:
  - Projected input: `gate, _ = self.g_proj(hidden_states)`.
  - Softplus: `gate = F.softplus(gate.float())`.
  - Application: `attn_output * gate`.
  - Support for `per-head` gating (shape `[num_heads]`) or `per-dim` gating (shape `[num_heads * head_dim]`).
- **Laguna-specific target projection**: `DFlashLagunaForCausalLM` (Lines 881-907) applies `aux_hidden_norms` (one per target layer) *before* concatenating and projecting via `fc`.

---

## 10. Weight Loading and Aliases

**`resolve_param_name` (Lines 739-754):**
Aliases for the checkpoint names:
- `encoder.fc.weight` $\to$ `fc.weight`
- `encoder.output_norm_enc.weight` $\to$ `hidden_norm.weight`
- Strips `model.` prefix if present.

**Stacked Parameters**:
- `q_proj`, `k_proj`, `v_proj` $\to$ `qkv_proj`
- `gate_proj`, `up_proj` $\to$ `gate_up_proj`
