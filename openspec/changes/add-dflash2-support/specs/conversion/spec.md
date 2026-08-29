# conversion — DFlash2 checkpoint to GGUF

## ADDED Requirements

### Requirement: REQ-CONV-1 GGUF conversion of DFlash2 checkpoint

The converter SHALL convert `incoai/GLM-5.3-Flash-DFlash2` (BF16 safetensors, 81 tensors) into a GGUF loadable by the glm5 fork's `llama-server` as a draft model.

- Vocab SHALL be taken from the GLM-5.3-Flash target GGUF/model dir (eagle3-style `set_vocab`; the draft checkpoint has none).
- The output GGUF SHALL carry metadata: `dflash.block_size=8`, `conv_kernel_size=2`, `conv_group_size=16`, `selector_rank=256`, `selector_top_k=16`, `target_layers=[6,15,25,34,43]`, `mask_token_id=154856`.
- `dflash.block_size` SHALL be 8 (the trained value), never the impl default of 16; model metadata SHALL always win.

#### Scenario: successful conversion

- **WHEN** the converter is run on `incoai/GLM-5.3-Flash-DFlash2` with the GLM-5.3-Flash target dir as vocab source
- **THEN** a GGUF is produced that the glm5 fork's `llama-server` loads as a draft model
- **AND** the GGUF metadata contains exactly the values listed above, including `dflash.block_size=8`

#### Scenario: arch name rejected → alias registered

- **WHEN** the converter rejects the `DFlash2DraftModel` arch name
- **THEN** the fix SHALL be to register the alias `@ModelBase.register("DFlash2DraftModel")` — the smallest possible patch, no new converter logic

### Requirement: REQ-CONV-2 Tensor inventory parity

The output GGUF SHALL contain exactly 81 tensors whose names and shapes match the rev-1 table (fc, enc_output_norm, blk.N.*, selector_*, etc. per the existing mapping).

#### Scenario: tensor count and shapes verified

- **WHEN** the converted GGUF is inspected
- **THEN** the tensor inventory (names + shapes) matches the rev-1 table at 81 tensors with no missing, extra, or mis-shaped tensors

### Requirement: REQ-CONV-3 Metadata parity vs reference GGUF

The output GGUF metadata SHALL be diffed against the known-good reference `incoai/Qwen3.8-27B-DFlash2-GGUF` (Q4_K_M, output of the same converter path); only model-specific differences SHALL be permitted.

#### Scenario: metadata mismatch detected → halt

- **WHEN** the metadata diff against the reference GGUF shows a difference that is not model-specific
- **THEN** the conversion is halted and the difference is investigated before any spec-decode work proceeds

### Requirement: REQ-CONV-4 Conv base layout golden check

The converter SHALL transpose `attn_conv_base` from the checkpoint layout `[2, 2, 4096]` (side, tap, channel) to the GGUF layout `[n_embd, kernel, 2]` (i.e. `[4096, 2, 2]`).

#### Scenario: wrong layer indexing caught by golden test

- **WHEN** the conv base transpose or the target-layer +1 indexing (HF 0-indexed `[5,14,24,33,42]` → GGUF 1-indexed `[6,15,25,34,43]`) is implemented incorrectly
- **THEN** the golden test on the conv base layout fails, catching the silent off-by-one that would otherwise produce plausible-but-garbage drafts
