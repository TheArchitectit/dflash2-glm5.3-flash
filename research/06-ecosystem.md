# DFlash2 Ecosystem Research Report

## 1. DFlash2 Blog Post Analysis (inco.ai/blog/dflash2/)

### Algorithm Description
DFlash2 introduces two key components to the original DFlash algorithm:

1. **Path Selector (~2.0M parameters)**:
   - Keeps top-16 candidates per position instead of top-1
   - Scores adjacent pairs via: S_t(a,b) = U_t(b) + ⟨A(a) ⊙ H(h_t), B(b)⟩
   - U_t(b): DFlash's logit for candidate b
   - A,B: compact 256-dim token embeddings
   - H(h_t): context gate (low-rank bilinear attention)
   - Final greedy walk picks coherent path; sampling preserves exact target distribution
   - Addresses oracle selection gap: 4.27 → 6.79 acceptance length (improves from 85.4% to 99.5% top-16 hit rate)

2. **Two-tap Dynamic Depthwise Convolution (~16.5M parameters)**:
   - Targets "suffix decay" in recall toward block end
   - Conv_k(x)_t = k_{t,0} ⊙ x_t + k_{t,1} ⊙ x_{t−1}
   - Each coefficient combines learned base kernel with hidden-state-derived correction
   - Inserted before/after each attention and feed-forward sublayer
   - Reduces within-block attention mass in layers 4-5 from 9.4% to 0.5%
   - Enables 5-layer model to nearly match 15-layer performance

### Training Procedure
- Minimal details provided
- States: "DFlash and DSpark baselines were 'trained... ourselves under matched setups, while MTP ships with the model'"
- No datasets, steps, hyperparameters, or loss functions described

### Benchmark Methodology
- **Metric**: per-request mean acceptance length (verifier-accepted tokens per cycle, including verifier's next token)
- **Also**: conditional acceptance rate at each draft position (conditioned on earlier positions being correct)
- **Benchmarks**: GSM8K, MATH-500, HumanEval, MBPP, MT-Bench
- **Sampling**: thinking enabled, temperature 1.0, top-p 0.95, top-k 20, presence penalty 1.5, with lossless rejection sampling
- **Block sizes**: 8 (Qwen3.8-27B), 16 (Muse Glimmer)
- **Ablations**: five-layer Qwen3-4B DFlash on GSM8K; comparisons include oracle top-16 selection, depth scaling

### Acceptance-Length Tables
**Qwen3.5-4B:**
| Dataset | MTP | DFlash | DSpark | DFlash 2 |
|---------|-----|--------|--------|----------|
| GSM8K | 4.78 | 4.99 | 5.69 | **6.20** |
| MATH-500 | 5.04 | 5.42 | 6.20 | **6.76** |
| HumanEval | 4.84 | 5.43 | 5.80 | **6.28** |
| MBPP | 4.16 | 4.49 | 4.96 | **5.41** |
| MT-Bench | 3.90 | 4.26 | 4.77 | **5.20** |
| **Mean** | 4.54 | 4.92 | 5.49 | **5.97** |

**Qwen3.8-27B (block size 8):**
| Dataset | MTP | DSpark | DFlash 2 |
|---------|-----|--------|----------|
| GSM8K | 5.02 | 4.36 | **5.46** |
| MATH-500 | 4.72 | 3.92 | **5.28** |
| HumanEval | 3.91 | 3.30 | **4.39** |
| MBPP | 3.99 | 3.51 | **4.79** |
| MT-Bench | 3.74 | 3.01 | **4.10** |
| **Mean** | 4.28 | 3.62 | **4.80** |

**Muse Glimmer (block size 16):**
| Dataset | DFlash | DSpark | DFlash 2 |
|---------|--------|--------|----------|
| GSM8K | 5.43 | 5.45 | **6.57** |
| MATH-500 | 5.39 | 5.01 | **6.56** |
| HumanEval | 4.11 | 4.33 | **5.66** |
| MBPP | 3.74 | 4.02 | **5.30** |
| MT-Bench | 3.52 | 3.59 | **4.42** |
| **Mean** | 4.44 | 4.48 | **5.70** |

### Throughput Numbers
- DFlash 2 on Qwen3.8-27B in SGLang: **2.7–3.4×** autoregressive decoding throughput
- On Muse Glimmer: **3.1–4.6×** throughput
- Per-token: close to 3× decode speed at ~1/3 compute, identical output
- Added cost: 1.3% draft–verify cycle latency for both components
- Legacy context: NVIDIA reported up to 15× on Blackwell; Google reported 3× on TPUs; >3.5M HF downloads (Aug 2026)

### llama.cpp / GGUF Support
- Already available via unmerged PR #27342
- Instructions: Clone llama.cpp, fetch `origin pull/27342/head:pr-27342`, build with CUDA/Metal
- Serve with: `-hf ggml-org/Qwen3.8-27B-GGUF:Q4_K_M -hfd incoai/Qwen3.8-27B-DFlash2-GGUF:Q4_K_M --spec-type draft-dflash --spec-draft-n-max 7`
- Q4_K_M GGUF drafter published: `incoai/Qwen3.8-27B-DFlash2-GGUF`
- No date/plan for upstream merge into mainline llama.cpp mentioned

## 2. Official DFlash GitHub Repository (github.com/z-lab/dflash)

### Repository Contents
- Python package (`dflash/` with `pyproject.toml`) for **inference and benchmarking**
- Block diffusion draft models for speculative decoding
- CLI with `generate` and `benchmark` subcommands across three backends:
  - Transformers (local, Linux)
  - MLX (Apple Silicon)
  - OpenAI-compatible server
- **No training code or model conversion tools** described
- Models: DFlash (for Muse-Glimmer-30B, Qwen3/Llama-3.1-8B) and DFlash 2

### GGUF Support
- Not mentioned directly
- However, llama.cpp listed as supported serving option (implies GGUF-format models)

### Issues/PRs mentioning llama.cpp
- No specific issues/PRs named
- README links to llama.cpp PR #27342 as part of serving support
- Repository shows 86 open issues, 12 PRs, and Discussions

### License
- MIT license

### Inference Frameworks Referenced
- **Transformers** (local, Linux) — DFlash 2 for Muse-Glimmer-30B, DFlash for Qwen3/Llama-3.1-8B
- **MLX** (Apple Silicon) — DFlash 2 for Qwen3.8-27B; DFlash for Qwen3/3.5/3.6 and Gemma 4
  - Note: quantized drafts should use "block_size <= 5" due to kernel efficiency limits
- **SGLang** — via PR #35371
- **vLLM** — via PR #52816
- **oMLX** — z-lab fork release download
- **llama.cpp** — via PR #27342
- For server-based options: run framework yourself, point CLI at endpoint via `--base-url`
- Model checkpoints hosted in Hugging Face collections: `z-lab/dflash` and `z-lab/dflash-2`

## 3. HuggingFace GGUF Conversions Search

Search attempts for "DFlash2 GGUF", "dflash gguf", "GLM-5.3-Flash draft" encountered API errors preventing direct results. However, based on the blog post and llama.cpp code analysis:

### Confirmed GGUF DFlash2 Models
- `incoai/Qwen3.8-27B-DFlash2-GGUF` (Q4_K_M quantization) - explicitly mentioned in blog post
- Available for use with llama.cpp draft-dflash spec type

### Search Limitations
Web search API encountered errors (502, 429, 400) when querying HuggingFace directly, preventing comprehensive enumeration of available GGUF conversions.

## 4. DFlash2 Drafts for Other Target Models

Based on available information:

### Confirmed Target Models
- **Primary**: Muse-Glimmer-30B (DFlash 2 specific)
- **Secondary**: Qwen3 series (Qwen3.5-4B, Qwen3.8-27B) - validated in benchmarks
- **Implied**: Llama-3.1-8B family (mentioned for original DFlash)

### Other Potential Targets
- **DeepSeek variants**: Referenced in research context (GLM-5.3-Flash CPU candidate, DeepSeek-V4 CPU decode collapse)
- **Nemotron**: SGLang code referenced `is_nemotron_35_draft` handling
- No specific DFlash2 drafts confirmed for these targets in available sources

## 5. SGLang PR 36708 Discussion Analysis

**PR**: Support GLM-5.3-Flash hidden-state capture (merged Aug 27, 2026)

### CPU Support
- **No mentions** of CPU support in discussion
- Entirely CUDA/GPU-focused (CUDA graphs, NVFP4 quantization, DGX Spark hardware)

### GGUF / llama.cpp
- **Not discussed** - work focused on GPU frameworks

### Ports to Other Frameworks
- Yes: vLLM port via sfxnz/GLM-5.3-Flash-NVFP4-vLLM-2x-DGX-Spark repo
- Technique credited to this PR: "capture completed layer output with SGLang-verified mHC hc_post + stream-average contraction"
- Addressed GLM-5-Next KV grouping issue by giving draft model own KV group at 1/4 attention block size

### DFlash2 Implementation Issues
1. Follow-up PR #36755: "Fix DFLASH aux hidden-state capture on mHC models" - indicated mHC contraction approach needed correction
2. Deployment pitfall: PR merged into feature branch from #36507 rather than main, causing pinned install image to lack capture
3. vLLM port documents: GLM-5-Next originally lacked EAGLE3-style interface DFlash uses to tap target hidden states

### Other Target Models
- **No mentions** of DeepSeek or Nemotron
- All discussion concerns GLM-5.3-Flash / GLM-5-Next (`Glm5NextForConditionalGeneration` architecture)

## 6. llama.cpp GitHub Issues/PRs Search for "dflash"

Based on direct repository examination:

### DFlash2 Support Status
- **Commit b10f9ca58** (Aug 27, 2026): "spec : add DFlash2 support (local convolution + candidate selector)" (#27342) (#27816)
- **Commit 4a6ad487a** (Aug 27, 2026): "spec : add DFlash2 support (local convolution + candidate selector)" (#27342)
- Both commits authored by Zihan Zhang and Xuan-Son Nguyen, assisted by Claude Opus 5
- Changes span 16 files: common/speculative.cpp, conversion/, ggml/src/ggml-cuda/top-k.cu, gguf-py/, src/, tests/

### Integration Status
- **MERGED into main/master branch** (verified via `git merge-base --is-ancestor b10f9ca58 origin/master`)
- DFlash2 support is already part of llama.cpp mainline as of Aug 27, 2026
- No open PRs/issues specifically for DFlash2 port - it's already implemented

### Related Commits
- DFlash support for Nemotron-3.5 (Aug 11, 2026): cc078b45b
- Muse Glimmer Support (Jun 24, 2026): 62bf73d25
- DSpark speculative decoding (Jan 19, 2026): 84075273c
- Auto-download dflash- and eagle3- HF sidecars (Apr 17, 2026): 635cdd5fc

### Code Changes Summary
- Added DFlash2 model architecture with:
  - Convolution layers (attention and FFN)
  - Selector networks (predecessor, successor, hidden)
  - GGUF tensor mapping for new components
  - Speculative sampling logic updates
  - Backend sampling enablement for both dflash & dspark
  - p_min support in DFlash2
  - Graph optimizations and bug fixes (mrope, lazy tensor, etc.)

## 7. Statements on llama.cpp Plans from incoai or z-lab

### From DFlash2 Blog Post (inco.ai)
- Explicitly states llama.cpp support is **already available** via PR #27342
- Provides specific usage instructions for testing the unmerged PR
- Confirms Q4_K_M GGUF drafter (`incoai/Qwen3.8-27B-DFlash2-GGUF`) is published
- **No statement** about future plans or timeline for upstream merge
- Focus is on enabling immediate use through the PR branch

### From z-lab/dflash Repository
- Lists llama.cpp as supported serving option (via PR #27342)
- No additional statements about plans or timelines
- Repository activity suggests maintenance focus on inference/benchmarking rather than framework integration

## Key Findings for llama.cpp Port Decision

### 1. **Port Already Completed**
- DFlash2 support is **already merged** into llama.cpp main/master branch (commit b10f9ca58/b10f9ca58)
- No work needed - the port exists and is usable
- Verified through direct git history analysis showing integration commits

### 2. **Official Documentation Available**
- Blog post provides complete algorithm description (path selector + dynamic convolution)
- Benchmark methodology and results clearly documented
- Acceptance-length tables provide validation targets
- Throughput numbers establish performance expectations

### 3. **GGUF Model Availability**
- Confirmed GGUF DFlash2 model: `incoai/Qwen3.8-27B-DFlash2-GGUF` (Q4_K_M)
- Enables immediate testing with llama.cpp using provided serve command
- No evidence of other DFlash2 GGUF conversions (DeepSeek variants, etc.) in accessible sources

### 4. **Validation Targets Available**
- Qwen3.5-4B and Qwen3.8-27B benchmarks provide clear validation targets
- Acceptance lengths: 5.97 mean (Qwen3.5-4B), 4.80 mean (Qwen3.8-27B block size 8)
- Throughput: 2.7-3.4× to 3.1-4.6× autoregressive decoding
- These targets exist for validation without needing DeepSeek-specific drafts

### 5. **No Active Conflicting Work**
- No open llama.cpp issues/PRs for DFlash2 (it's already merged)
- No evidence of competing port efforts in progress
- SGLang work is GPU-focused and unrelated to CPU/llama.cpp port

### Recommendation
**Proceed with validation/testing rather than development:**
1. Use existing llama.cpp main branch (includes DFlash2 support)
2. Download the confirmed GGUF model: `incoai/Qwen3.8-27B-DFlash2-GGUF`
3. Validate against published benchmarks (Qwen3.5-4B/Qwen3.8-27B acceptance lengths)
4. If DeepSeek validation is required, note that algorithm is target-agnostic and can work with any compatible draft model

The ecosystem research shows the technical work is complete; focus should shift to validation and optimization within the llama.cpp context.
