#!/usr/bin/env python3
"""Sprint 3.4 helper — build a minimal mock target GGUF for the replay harness.

The dflash draft GGUF is headless (borrows target tok_embd + output via
ctx_other). dflash.cpp only reads model_other->tok_embd and model_other->output
— it does NOT require a glm5next arch. So a trivial llama-arch mock carrying
the shared fixture matrix (fixtures/lm_head.npy, seed 123) keeps both arms
comparable without reconstructing all of glm5next's hparams.

Writes tests/golden/fixtures/mock_target.gguf:
  token_embd.weight [4096, 154880] F16, output.weight [4096, 154880] F16.
"""

import os
import sys

import numpy as np

sys.path.insert(0, "/mnt/ollama/models/llama-cpp-glm5/gguf-py")
import gguf  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.join(HERE, "fixtures")
OUT = os.path.join(FIXTURES, "mock_target.gguf")
N_EMBD = 4096
VOCAB = 154880


def main():
    emb = np.load(os.path.join(FIXTURES, "lm_head.npy"))
    assert emb.shape == (VOCAB, N_EMBD), emb.shape
    emb_f16 = emb.astype(np.float16)

    # copy the draft's tokenizer verbatim (identical vocab ids)
    r = gguf.GGUFReader('/mnt/ollama/models/glm-5.3-flash/dflash2-gguf/dflash2-glm-f16.gguf')
    f = {str(i.name): i for i in r.fields.values()}
    toks = f['tokenizer.ggml.tokens'].contents()
    typs = f['tokenizer.ggml.token_type'].contents()
    merges = f['tokenizer.ggml.merges'].contents()

    w = gguf.GGUFWriter(OUT + ".tmp", arch="llama")
    w.add_tokenizer_model("gpt2")
    w.add_tokenizer_pre("glm4")
    w.add_token_list(toks)
    w.add_token_types([int(t) for t in typs])
    w.add_token_merges(merges)
    w.add_name("golden-mock-target")
    w.add_context_length(4096)
    w.add_embedding_length(N_EMBD)
    w.add_block_count(1)
    w.add_feed_forward_length(1)
    w.add_head_count(32)
    w.add_head_count_kv(32)
    w.add_layer_norm_rms_eps(1e-5)
    w.add_rope_dimension_count(128)
    w.add_bos_token_id(f['tokenizer.ggml.bos_token_id'].contents())
    w.add_eos_token_id(f['tokenizer.ggml.eos_token_id'].contents())
    w.add_pad_token_id(f['tokenizer.ggml.padding_token_id'].contents())
    w.add_mask_token_id(f['tokenizer.ggml.mask_token_id'].contents())
    w.add_tensor("token_embd.weight", emb_f16)
    w.add_tensor("output_norm.weight", np.ones(N_EMBD, dtype=np.float16))
    # block 0 stubs so the trivial llama arch loads (unused by the dflash draft;
    # ctx_other only provides tok_embd/output)
    w.add_tensor("blk.0.attn_norm.weight", np.ones(N_EMBD, dtype=np.float16))
    w.add_tensor("blk.0.attn_q.weight", np.zeros((N_EMBD, N_EMBD), dtype=np.float16))
    w.add_tensor("blk.0.attn_k.weight", np.ones((N_EMBD, N_EMBD), dtype=np.float16))
    w.add_tensor("blk.0.attn_v.weight", np.ones((N_EMBD, N_EMBD), dtype=np.float16))
    w.add_tensor("blk.0.attn_output.weight", np.ones((N_EMBD, N_EMBD), dtype=np.float16))
    w.add_tensor("blk.0.ffn_gate.weight", np.ones((1, N_EMBD), dtype=np.float16))
    w.add_tensor("blk.0.ffn_up.weight", np.ones((1, N_EMBD), dtype=np.float16))
    w.add_tensor("blk.0.ffn_down.weight", np.ones((N_EMBD, 1), dtype=np.float16))
    w.add_tensor("blk.0.ffn_norm.weight", np.ones(N_EMBD, dtype=np.float16))
    w.add_tensor("output.weight", emb_f16)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    os.replace(OUT + ".tmp", OUT)
    print(f"wrote {OUT} ({os.path.getsize(OUT) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
