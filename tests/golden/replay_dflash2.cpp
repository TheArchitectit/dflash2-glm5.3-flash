// Sprint 3.4 — llama.cpp replay of the DFlash2 draft on the 3.1 fixture.
//
// Draft-only (no 147 GB target load): loads dflash2-glm-f16.gguf, feeds the
// SAME fixture hiddens through the production code path:
//   1. project_target_hidden   (fc → hidden_norm)        — encoder graph
//   2. KV materialization      (embeds decode)            — production path
//   3. noise block decode      ([anchor, mask×7])
//   4. selector lattice walk   (speculative.cpp:1265-1285 logic, greedy)
// and dumps the same intermediates the SGLang arm (3.3) dumped, for the
// 1e-3 comparison in 3.5.
//
// usage: replay_dflash2 -md <dflash2.gguf>
//        env: GOLD_FIXTURE=<hiddens.bin> GOLD_OUT=<replay.bin>

#include "common.h"
#include "arg.h"
#include "llama.h"
#include "llama-ext.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static const int BLOCK_SIZE = 8;

int main(int argc, char ** argv) {
    common_params params;
    // parser requires a --model even for draft-only runs; point it at the
    // draft GGUF (unused — we load the draft explicitly below).
    params.model.path = params.speculative.draft.mparams.path;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SPECULATIVE)) {
        return 1;
    }
    params.model.path = "";  // never actually load the target

    std::string fixture_path = "tests/golden/fixtures/hiddens.bin";
    std::string out_path     = "tests/golden/fixtures/llamacpp_replay.bin";
    if (const char * e = getenv("GOLD_FIXTURE")) fixture_path = e;
    if (const char * e = getenv("GOLD_OUT"))     out_path     = e;

    if (params.speculative.draft.mparams.path.empty()) {
        fprintf(stderr, "usage: replay_dflash2 -md <dflash2.gguf>  "
                        "(env GOLD_FIXTURE=<hiddens.bin> GOLD_OUT=<replay.bin>)\n");
        return 1;
    }

    // ---- load fixture (HID1 format from dump_target_hiddens) -------------
    FILE * ff = fopen(fixture_path.c_str(), "rb");
    if (!ff) { fprintf(stderr, "cannot open fixture %s\n", fixture_path.c_str()); return 1; }
    uint32_t magic = 0;
    int32_t hdr[3] = {};
    if (fread(&magic, 4, 1, ff) != 1 || magic != 0x48494431) { fprintf(stderr, "bad fixture magic\n"); return 1; }
    if (fread(hdr, 4, 3, ff) != 3) { fprintf(stderr, "bad fixture header\n"); return 1; }
    const int32_t n_tok = hdr[0], n_embd = hdr[1];
    const uint32_t n_layers = (uint32_t) hdr[2];
    std::vector<llama_token> fx_tokens(n_tok);
    if (fread(fx_tokens.data(), 4, n_tok, ff) != (size_t) n_tok) { fprintf(stderr, "bad fixture tokens\n"); return 1; }
    std::vector<float> fx_layers((size_t) n_tok * n_layers * n_embd);
    for (uint32_t l = 0; l < n_layers; ++l) {
        if (fread(fx_layers.data() + (size_t) l * n_tok * n_embd, 4, (size_t) n_tok * n_embd, ff)
                != (size_t) n_tok * n_embd) { fprintf(stderr, "bad fixture layer %u\n", l); return 1; }
    }
    fclose(ff);
    printf("fixture: %d tokens, %u layers x %d dims\n", n_tok, n_layers, n_embd);
    // HID1 stores layer-major [layer][token][embd]; the encoder batch wants
    // token-major [token][layer*n_embd] (production features_buf layout)
    std::vector<float> features((size_t) n_tok * n_layers * n_embd);
    for (uint32_t l = 0; l < n_layers; ++l) {
        for (int32_t i = 0; i < n_tok; ++i) {
            std::memcpy(features.data() + (size_t) i * n_layers * n_embd + (size_t) l * n_embd,
                        fx_layers.data() + ((size_t) l * n_tok + i) * n_embd,
                        (size_t) n_embd * sizeof(float));
        }
    }

    // ---- init draft model + mock target (headless draft borrows its
    //      tok_embd/output via ctx_other; the mock carries the shared
    //      fixture matrix so both arms see identical embeddings/lm_head) ----
    llama_backend_init();

    const char * mock_path = getenv("GOLD_MOCK_TARGET");
    std::string mock = mock_path ? mock_path : "tests/golden/fixtures/mock_target.gguf";
    auto tmparams = common_model_params_to_llama(params);
    llama_model * model_other = llama_model_load_from_file(mock.c_str(), tmparams);
    if (!model_other) { fprintf(stderr, "failed to load mock target\n"); return 1; }
    auto tcparams = common_context_params_to_llama(params);
    tcparams.n_ctx     = 64;
    tcparams.n_batch   = 64;
    tcparams.n_ubatch  = 64;
    llama_context * ctx_other = llama_init_from_model(model_other, tcparams);
    if (!ctx_other) { fprintf(stderr, "failed to create mock target context\n"); return 1; }

    auto mparams = common_model_params_to_llama(params);
    llama_model * model = llama_model_load_from_file(params.speculative.draft.mparams.path.c_str(), mparams);
    if (!model) { fprintf(stderr, "failed to load draft model\n"); return 1; }
    auto cparams = common_context_params_to_llama(params);
    cparams.n_ctx     = (uint32_t) (n_tok + BLOCK_SIZE + 8);
    cparams.n_batch   = cparams.n_ctx;
    cparams.n_ubatch  = cparams.n_batch;
    cparams.n_threads = 20;
    cparams.ctx_other = ctx_other;  // supplies tok_embd + output for the headless draft
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) { fprintf(stderr, "failed to create draft context\n"); return 1; }
    // production dflash2 setup (speculative.cpp:1054): unmasked nextn —
    // one embedding row per token, the selector lattice among them
    llama_set_embeddings_nextn(ctx, true, /*masked*/ false);
    llama_set_causal_attn(ctx, false);

    const int32_t  n_embd_dec = llama_model_n_embd(model);
    const uint32_t layer_n    = llama_model_target_layer_ids_n(model);
    if ((int) layer_n != n_layers || n_embd != llama_model_n_embd(model)) {
        fprintf(stderr, "fixture dims (%u x %d) do not match model (%u x %d)\n",
                n_layers, n_embd, layer_n, llama_model_n_embd(model));
        return 1;
    }

    // ---- 1+2. encoder + KV materialization (production process() path) --
    llama_batch enc_batch = llama_batch_init(n_tok, n_layers * n_embd, 1);
    enc_batch.n_tokens = n_tok;
    std::memcpy(enc_batch.embd, features.data(), features.size() * sizeof(float));
    for (int32_t i = 0; i < n_tok; ++i) {
        enc_batch.pos[i]      = i;
        enc_batch.n_seq_id[i] = 1;
        enc_batch.seq_id[i][0] = 0;
        enc_batch.logits[i]   = false;
    }
    if (llama_encode(ctx, enc_batch) != 0) {
        fprintf(stderr, "llama_encode failed\n");
        return 1;
    }
    const float * inp_g = llama_get_embeddings_nextn(ctx);
    if (!inp_g) { fprintf(stderr, "encoder produced no output\n"); return 1; }
    std::vector<float> ctx_hidden((size_t) n_tok * n_embd_dec);
    std::memcpy(ctx_hidden.data(), inp_g, ctx_hidden.size() * sizeof(float));
    llama_batch_free(enc_batch);

    // KV materialization at the ctx positions (batch_inject path)
    llama_batch binj = llama_batch_init(n_tok, n_embd_dec, 1);
    binj.n_tokens = n_tok;
    std::memcpy(binj.embd, ctx_hidden.data(), ctx_hidden.size() * sizeof(float));
    for (int32_t i = 0; i < n_tok; ++i) {
        binj.pos[i]      = i;
        binj.n_seq_id[i] = 1;
        binj.seq_id[i][0] = 0;
        binj.logits[i]   = false;
    }
    if (llama_decode(ctx, binj) != 0) {
        fprintf(stderr, "KV materialization decode failed\n");
        return 1;
    }
    llama_batch_free(binj);

    // ---- 3. noise block decode -------------------------------------------
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const llama_token mask_id = llama_vocab_mask(vocab);
    const llama_token anchor  = fx_tokens[n_tok - 1];
    llama_batch blk = llama_batch_init(BLOCK_SIZE, 0, 1);
    blk.n_tokens = BLOCK_SIZE;
    for (int32_t i = 0; i < BLOCK_SIZE; ++i) {
        blk.token[i]  = (i == 0) ? anchor : mask_id;
        blk.pos[i]    = n_tok + i;  // production: n_past = N, block at N..N+7
        blk.n_seq_id[i] = 1;
        blk.seq_id[i][0] = 0;
        blk.logits[i] = false; // dflash2 reads the lattice from nextn embeddings
    }
    if (llama_decode(ctx, blk) != 0) {
        fprintf(stderr, "noise block decode failed\n");
        return 1;
    }
    llama_batch_free(blk);

    // ---- 4. selector lattice + greedy walk (speculative.cpp:1265-1285) ---
    const float * lattice = llama_get_embeddings_nextn(ctx);
    if (!lattice) { fprintf(stderr, "selector produced no lattice\n"); return 1; }
    const int32_t top_k = llama_model_dflash_selector_top_k(model);
    if (top_k <= 0) { fprintf(stderr, "not a dflash2 model (no selector)\n"); return 1; }
    const int32_t n_block_tokens = BLOCK_SIZE; // [anchor + 7 predictions]

    std::vector<llama_token> proposed;
    std::vector<float> scores_rows; // [n_slots * top_k] the argmax-walked rows
    int32_t predecessor = 0;        // production walk: lattice row starts at 0
    for (int32_t i = 1; i < n_block_tokens; ++i) {
        const float * row    = lattice + (size_t) i * n_embd_dec;
        const float * scores = row + top_k + (size_t) predecessor * top_k;
        predecessor = (int32_t) std::distance(scores,
                std::max_element(scores, scores + top_k));
        scores_rows.insert(scores_rows.end(), scores, scores + top_k);
        proposed.push_back((llama_token) row[predecessor]);
    }

    printf("proposed %zu tokens:", proposed.size());
    for (auto t : proposed) printf(" %d", (int) t);
    printf("\n");

    // ---- dump intermediates (consumed by compare_golden.py) --------------
    FILE * out = fopen(out_path.c_str(), "wb");
    if (!out) { fprintf(stderr, "cannot open %s\n", out_path.c_str()); return 1; }
    const uint32_t omagic = 0x52454B31; // "REK1"
    int32_t n_prop = (int32_t) proposed.size();
    int32_t n_slot_rows = n_block_tokens;
    fwrite(&omagic, 4, 1, out);
    fwrite(&n_embd_dec, 4, 1, out);
    fwrite(&top_k, 4, 1, out);
    fwrite(&n_slot_rows, 4, 1, out);
    fwrite(&n_prop, 4, 1, out);
    fwrite(proposed.data(), 4, proposed.size(), out);
    fwrite(scores_rows.data(), 4, scores_rows.size(), out);
    // full lattice [n_block_tokens, n_embd_dec] — rows 0..7: [cand ids | scores | ...]
    fwrite(lattice, sizeof(float), (size_t) n_block_tokens * n_embd_dec, out);
    // ctx_hidden [n_tok, n_embd_dec] — the encoder output (fc → hidden_norm),
    // directly comparable to sglang_golden's ctx_hidden
    fwrite(ctx_hidden.data(), sizeof(float), ctx_hidden.size(), out);
    fclose(out);
    printf("wrote %s (n_embd_dec=%d top_k=%d)\n", out_path.c_str(), n_embd_dec, top_k);

    llama_free(ctx);
    llama_free(ctx_other);
    llama_model_free(model);
    llama_model_free(model_other);
    llama_backend_free();
    return 0;
}
