// Sprint 3.1 — dump per-layer target hiddens for the golden test.
//
// Loads the GLM-5.3-Flash IQ4_XS target, enables layer-input extraction for
// the 5 dflash target layers (1-indexed in llama.cpp: {6,15,25,34,43} =
// 0-indexed {5,14,24,33,42}), runs ONE prefill decode on a canned agentic
// prompt, and writes the extracted hiddens + token ids to a raw binary
// fixture (converted to npz by tests/golden/save_npz.py).
//
// This is the deployment-realistic hidden source: same quantization, same
// extraction path (llama_get_embeddings_layer_inp, post build_hc_mean) the
// draft sees in production.
//
// usage: dump_target_hiddens -m <target.gguf> -o <out.bin> [--prompt-file f]
//
// NOTE: 147 GB model load — run solo per the 2.1 discipline (no :8086/:8100).

#include "common.h"
#include "arg.h"
#include "llama.h"
#include "llama-ext.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

// llama.cpp metadata is 1-indexed: {6,15,25,34,43}
static const std::vector<uint32_t> TARGET_LAYERS = {6, 15, 25, 34, 43};

// Canned agentic prompt (~40 tokens): a tool-calling request.
static const std::string DEFAULT_PROMPT =
    "You are a helpful assistant with access to tools. "
    "Search the web for the capital of France, then call get_weather for that city "
    "and summarize the result in one sentence.";

int main(int argc, char ** argv) {
    common_params params;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    std::string out_path = "hiddens.bin";
    std::string prompt = DEFAULT_PROMPT;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "-o" && i + 1 < argc) { out_path = argv[++i]; }
        if (a == "--prompt-file" && i + 1 < argc) {
            FILE * f = fopen(argv[++i], "r");
            if (!f) { fprintf(stderr, "cannot open prompt file\n"); return 1; }
            prompt.clear();
            char buf[4096];
            size_t n;
            while ((n = fread(buf, 1, sizeof(buf), f)) > 0) prompt.append(buf, n);
            fclose(f);
        }
    }

    params.n_ctx = 512;
    params.n_batch = 512;
    params.n_ubatch = 512;

    auto mparams = common_model_params_to_llama(params);
    llama_model * model = llama_model_load_from_file(params.model.path.c_str(), mparams);
    if (!model) { fprintf(stderr, "failed to load model\n"); return 1; }

    auto cparams = common_context_params_to_llama(params);
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) { fprintf(stderr, "failed to create context\n"); return 1; }

    const int32_t n_embd = llama_model_n_embd(model);

    // tokenize (BOS if the model expects one)
    const bool add_bos = llama_vocab_get_add_bos(llama_model_get_vocab(model));
    std::vector<llama_token> tokens = common_tokenize(llama_model_get_vocab(model), prompt, add_bos, false);
    const int32_t n_tokens = (int32_t) tokens.size();
    printf("prompt: %d tokens (add_bos=%d)\n", n_tokens, (int) add_bos);

    // enable extraction for the 5 target layers
    for (uint32_t il : TARGET_LAYERS) {
        llama_set_embeddings_layer_inp(ctx, il, true);
    }

    // one prefill decode
    llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    for (int32_t i = 0; i < n_tokens; ++i) {
        common_batch_add(batch, tokens[i], i, {0}, false);
    }
    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "prefill decode failed\n");
        return 1;
    }
    llama_batch_free(batch);

    // dump: header = magic, n_tokens, n_embd, n_layers; then tokens, then layers in order
    FILE * out = fopen(out_path.c_str(), "wb");
    if (!out) { fprintf(stderr, "cannot open %s\n", out_path.c_str()); return 1; }
    const uint32_t magic = 0x48494431; // "HID1"
    const uint32_t n_layers = (uint32_t) TARGET_LAYERS.size();
    fwrite(&magic, 4, 1, out);
    fwrite(&n_tokens, 4, 1, out);
    fwrite(&n_embd, 4, 1, out);
    fwrite(&n_layers, 4, 1, out);
    fwrite(tokens.data(), sizeof(llama_token), n_tokens, out);
    for (uint32_t il : TARGET_LAYERS) {
        const float * emb = llama_get_embeddings_layer_inp(ctx, il);
        if (!emb) { fprintf(stderr, "layer %u not extracted\n", il); return 1; }
        fwrite(emb, sizeof(float), (size_t) n_tokens * n_embd, out);
        printf("layer %u: extracted %d x %d\n", il, n_tokens, n_embd);
    }
    fclose(out);
    printf("wrote %s\n", out_path.c_str());

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
