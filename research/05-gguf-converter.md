# GGUF Converter Implementation Guide for DFlash2

This document details the requirements and implementation patterns for adding a `safetensors`-to-GGUF converter for the `dflash2` architecture, based on existing implementations in `llama.cpp`.

## 1. Core Files to Modify

To support a new architecture, the following files must be updated:

### A. Converter (Python)
- **`conversion/base.py` / `conversion/__init__.py`**: Ensure `ModelBase` registration is handled.
- **`conversion/<arch>.py`**: Create a new model class (e.g., `DFlash2Model`) inheriting from `TextModel` or `LlamaModel`.
- **`gguf-py/gguf/gguf_writer.py`**: Add helper methods for architecture-specific metadata (e.g., `add_block_size`).
- **`convert_hf_to_gguf.py`**: Register the model class if not using decorators.

### B. C++ Backend (llama.cpp)
- **`src/llama-arch.h`**:
    - Add architecture enum: `LLM_ARCH_DFLASH2`.
    - Add tensor enums: `LLM_TENSOR_DFLASH2_*`.
    - Add metadata keys: `LLM_KV_DFLASH2_*`.
- **`src/llama-arch.cpp`**:
    - Map `LLM_ARCH_DFLASH2` to string `"dflash2"`.
    - Map `LLM_TENSOR_DFLASH2_*` to GGUF tensor names (e.g., `"blk.%d.attn_conv_base"`).
- **`src/llama-model.cpp`** & **`src/models/dflash2.cpp`**:
    - Implement `load_arch_hparams` to read metadata.
    - Implement `load_arch_tensors` to allocate tensors based on hparams.

---

## 2. Converter Implementation Patterns

### Architecture Registration
The converter uses a registration decorator. The new model class must specify its `model_arch`.

```python
@ModelBase.register(
    "DFlash2ForCausalLM",
    "DFlash2DraftModel",
)
class DFlash2Model(TextModel):
    model_arch = gguf.MODEL_ARCH.DFLASH  # Use existing DFLASH or new DFLASH2
```

### Handling Missing Embeddings/LM Head
Draft models often lack their own embedding tables or LM heads, relying instead on a target model.

1.  **Skip Tensors**: In `modify_tensors`, check for the existence of weights in `hparams` or skip specific names.
    ```python
    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if name == "model.embed_tokens.weight" and not self.hparams.get("has_embed_tokens", True):
            return # Skip writing this tensor
        yield from super().modify_tensors(data_torch, name, bid)
    ```
2.  **Vocab Delegation**: Use `--target-model-dir` to copy the tokenizer from the target model.
    ```python
    def set_vocab(self):
        if self.target_model_dir is not None:
            logger.info(f"Using tokenizer from target model: {self.target_model_dir}")
            self.dir_model = self.target_model_dir
            # ... call tokenizer loading logic ...
    ```

### Writing Custom Metadata
Metadata is written via the `gguf_writer`.

**In `gguf-py/gguf/gguf_writer.py`**:
```python
def add_block_size(self, value: int) -> None:
    self.add_key("dflash.block_size", gguf.GGMLType.I32, value)

def add_target_layers(self, value: Sequence[int]) -> None:
    self.add_key("target_layers", gguf.GGMLType.I32, value)
```

**In `conversion/<arch>.py`**:
```python
def set_gguf_parameters(self):
    super().set_gguf_parameters()
    dflash_config = self.hparams.get("dflash_config", {})
    self.gguf_writer.add_block_size(dflash_config.get("block_size", 16))
    
    target_layers = dflash_config.get("target_layer_ids", [])
    if target_layers:
        # Convert 0-indexed HF layers to 1-indexed GGUF if required
        self.gguf_writer.add_target_layers([i + 1 for i in target_layers])
```

---

## 3. Tensor Mapping (DFlash / DSpark)

The following tensor enums are used for DFlash-style architectures:

| C++ Enum | GGUF Name / Mapping |
| :--- | :--- |
| `LLM_TENSOR_DFLASH_ATTN_CONV_BASE` | `"blk.%d.attn_conv_base"` |
| `LLM_TENSOR_DFLASH_ATTN_CONV_PROJ` | `"blk.%d.attn_conv_proj"` |
| `LLM_TENSOR_DFLASH_FFN_CONV_BASE` | `"blk.%d.ffn_conv_base"` |
| `LLM_TENSOR_DFLASH_FFN_CONV_PROJ` | `"blk.%d.ffn_conv_proj"` |
| `LLM_TENSOR_DFLASH_SELECTOR_PREV` | `"selector_predecessor"` |
| `LLM_TENSOR_DFLASH_SELECTOR_NEXT` | `"selector_successor"` |
| `LLM_TENSOR_DFLASH_SELECTOR_HIDDEN` | `"selector_hidden"` |
| `LLM_TENSOR_DSPARK_MARKOV_W1` | `"markov_w1"` |
| `LLM_TENSOR_DSPARK_MARKOV_W2` | `"markov_w2"` |
| `LLM_TENSOR_DSPARK_CONF_PROJ` | `"conf_proj"` |

---

## 4. Technical Considerations

### Int64 Tensors (e.g., `d2t`)
Tensors requiring `int64` must bypass the standard F32 conversion in `prepare_tensors`.

```python
def prepare_tensors(self):
    # Collect original dtypes for int tensors
    int_tensors = {}
    for name, data_torch in self.get_tensors():
        if name == "d2t":
            int_tensors[name] = data_torch

    super().prepare_tensors()

    # Write explicitly as I64
    for name, data_torch in int_tensors.items():
        data = data_torch.to(torch.int64).cpu().numpy()
        self.gguf_writer.add_tensor(name, data, raw_dtype=gguf.GGMLQuantizationType.I64)
```

### Quantization and Splitting
- **`llama-gguf-split`**: Handles files based on the total byte size of tensors. If `token_embd` is missing, the first split simply starts with the first available tensor.
- **`quantize`**: Processes tensors by their GGUF type. Tensors without associated weights (like missing embeddings) are simply not present in the file and thus not quantized.
