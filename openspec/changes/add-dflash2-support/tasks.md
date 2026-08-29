# Tasks: add-dflash2-support

## Ground truth (verified on box)

| Item | Value |
|---|---|
| Fork | `/mnt/ollama/models/llama-cpp-glm5` (commit f30bed8, built; `build/bin/llama-server`) |
| Converter | `/mnt/ollama/models/llama-cpp-glm5/convert_hf_to_gguf.py` + `conversion/qwen.py` (`DFlashModel`, registers `DFlash2DraftModel` at qwen.py:642) |
| Checkpoint | `/mnt/ollama/models/glm-5.3-flash/dflash2/` (BF16, 81 tensors, 2.2 GB; `dflash_config`: block_size=8, kernel=2, group=16, rank=256, top_k=16, mask=154856, target_layer_ids=[5,14,24,33,42] 0-idx) |
| Target GGUF | `/mnt/ollama/models/glm-5.3-flash/UD-IQ4_XS/GLM-5.3-Flash-UD-IQ4_XS-0000{1..5}-of-00005.gguf` (147 GB) |
| SGLang ref impl | `/mnt/ollama/models/glm-5.3-flash/sglang-venv/lib/python3.12/site-packages/sglang/srt/models/dflash.py` (venv python: torch 2.13.0+cu130, transformers 5.16.1) |
| Reference GGUF | `incoai/Qwen3.8-27B-DFlash2-GGUF` on HF (81 tensors; parsed metadata: `dflash.block_size=8`, `target_layers=[6,20,34,48,62]` 1-idx, `causal=0`, `sliding_window=2048`, pattern `[1,1,1,1,1]`, `mask_token_id=248070`; `attn_conv_base` ne=[5120,2,2] F32) |
| Draft GGUF out | `/mnt/ollama/models/glm-5.3-flash/dflash2-gguf/dflash2-glm-f16.gguf` |
| Production | `llama-server-qwen-flash.service` on :8086 — ACTIVE. All server runs in Sprints 2–4 are solo: :8086 idle/stopped, never concurrent |
| Free port | :8100 (prior glm5 service port; `llama-server-glm5.service` inactive) |
| Expected GGUF | 81 tensors, no `token_embd`/`output` (headless draft; borrows target embeddings + lm_head); `fc` ne=[20480,4096]; `attn_conv_base` ne=[4096,2,2]; `selector_predecessor`/`selector_successor` ne=[256,154880]; `selector_hidden` ne=[4096,256] |

Checkpoint tensor inventory (ground truth for 1.6): 5 layers × 15 per-layer tensors (`input_layernorm`, `self_attn.{q,k,v,o}_proj`, `self_attn.{q,k}_norm`, `post_attention_layernorm`, `mlp.{gate,up,down}_proj`, `attention_conv.{base_kernel,kernel_projection}`, `mlp_conv.{base_kernel,kernel_projection}`) = 75, plus top-level `norm.weight`, `fc.weight` [4096,20480], `hidden_norm.weight` [4096], `candidate_selector.{hidden_projection [256,4096], predecessor_codebook [154880,256], successor_codebook [154880,256]}` = 6. Total 81.

---

## Sprint 1 — Conversion (Phase A, REQ-1) — COMPLETE 2026-08-29

All gates passed (81-tensor inventory, metadata parity vs reference, conv-base
byte-identical, llama-cli load sanity, off-by-one layers correct). Arch alias
already registered at qwen.py:642 — no patch needed.

- [ ] **1.1 Stage GLM-5.3-Flash target HF files.** The draft checkpoint has no tokenizer; `DFlashModel.set_vocab` (qwen.py:647-673) delegates to the target arch's vocab handler and needs real files, not the GGUF shards. Run:
  `hf download zai-org/GLM-5.3-Flash config.json tokenizer.json tokenizer_config.json generation_config.json chat_template.jinja --local-dir /mnt/ollama/models/glm-5.3-flash/target-hf`
  Expected: 5 small files, no safetensors (~30 MB). Gate: `config.json` contains `"architectures": ["Glm5NextForConditionalGeneration"]` (resolves to `Glm5NextModel` in `conversion/__init__.py:107` → `_set_vocab_glm` at conversion/base.py:2170).

- [ ] **1.2 Download reference GGUF for parity diff.** `hf download incoai/Qwen3.8-27B-DFlash2-GGUF Qwen3.8-27B-DFlash2-Q8_0.gguf --local-dir /mnt/ollama/models/glm-5.3-flash/ref-gguf` (2.1 GB; disk has 180 GB free). Expected outcome: known-good output of the same converter path, same `LLM_ARCH_DFLASH`, for 1.7/1.8. Gate: `python3 -c` GGUF header parse reports `tensors=81, kvs=47` (already verified via HTTP range request; re-verify locally after download).

- [ ] **1.3 Record checkpoint baseline inventory.** Write `scripts/check_tensor_inventory.py`: read `/mnt/ollama/models/glm-5.3-flash/dflash2/model.safetensors` header (81 tensors — names/shapes listed above), read converted GGUF via fork's `gguf-py`, assert 1:1 name+shape match through the torch→ggml dimension reversal (torch `[a,b,c]` → ggml ne `[c,b,a]`). Expected: 81 == 81, no `token_embd`, no `output.weight`, no `d2t` (dflash.cpp:99 marks tok_embd `TENSOR_NOT_REQUIRED`). Gate: script exit 0. **STOP GATE**: any missing/extra/mis-shaped tensor halts Sprint 1 (REQ-CONV-2).

- [ ] **1.4 Verify arch registration before burning a conversion.** `cd /mnt/ollama/models/llama-cpp-glm5 && python3 convert_hf_to_gguf.py --print-supported-models | grep -i dflash` and confirm `DFlash2DraftModel` is accepted (`conversion/__init__.py:57` maps it to module `qwen`). Contingency (spec D7, expected unneeded): if rejected, add the name to `@ModelBase.register(...)` at `conversion/qwen.py:642` — 1-line patch, no new logic. Gate: registration confirmed or alias patch committed.

- [ ] **1.5 Run the converter.**
  ```
  cd /mnt/ollama/models/llama-cpp-glm5
  python3 convert_hf_to_gguf.py \
    /mnt/ollama/models/glm-5.3-flash/dflash2 \
    --target-model-dir /mnt/ollama/models/glm-5.3-flash/target-hf \
    --outtype f16 \
    --outfile /mnt/ollama/models/glm-5.3-flash/dflash2-gguf/dflash2-glm-f16.gguf
  ```
  (F16 over BF16: safest CPU compute path for the 2.2 GB draft; `--outtype bf16` acceptable alternative.) Expected: single-file GGUF ~4.4 GB, no vocab errors, split not needed. Gate: file exists, `gguf` reader opens it, arch = `dflash`.

- [ ] **1.6 Tensor inventory check.** Run `scripts/check_tensor_inventory.py /mnt/ollama/models/glm-5.3-flash/dflash2-gguf/dflash2-glm-f16.gguf` (from 1.3). Expected: 81 tensors; `fc` ne=[20480,4096]; `enc.output_norm` ne=[4096]; selector trio shapes as in the ground-truth table. Gate: exit 0. **STOP GATE** on mismatch.

- [ ] **1.7 Metadata parity diff vs reference GGUF.** Write `scripts/diff_gguf_meta.py` (KV-dict diff of two GGUFs). Diff `dflash2-glm-f16.gguf` vs `ref-gguf/Qwen3.8-27B-DFlash2-Q8_0.gguf`. Expected: identical `dflash.block_size=8`, `dflash.conv_kernel_size=2`, `dflash.conv_group_size=16`, `dflash.selector_rank=256`, `dflash.selector_top_k=16`, `dflash.attention.causal=0`, `dflash.attention.sliding_window=2048`, `sliding_window_pattern=[1,1,1,1,1]`, `block_count=5`, `attention.head_count=32`, `head_count_kv=8`, `key_length=value_length=128`, `sliding_window=2048`. Allowed model-specific diffs (allowlist in script): `embedding_length` (4096 vs 5120), `feed_forward_length` (12288 vs 17408), `context_length` (1048576 vs 262144), `rope.freq_base` (10000 vs 1e7), all `tokenizer.*` (154880 vs 248320 vocab, `mask_token_id=154856` vs 248070, `pre`/merges), `general.*` (name/license `cc-by-nc-nd-4.0` vs `apache-2.0`, base_model, size_label). Gate: diff shows only allowlisted keys. **STOP GATE** (REQ-CONV-3): any non-allowlisted difference halts Sprint 1 — investigate before spec-decode work.

- [ ] **1.8 Layer off-by-one check (risk 2, the silent killer).** In the 1.7 diff, assert `dflash.target_layers == [6,15,25,34,43]` (converter applies +1 to HF `[5,14,24,33,42]`, qwen.py:707-710; reference shows `[6,20,34,48,62]` = 1-indexed for Qwen). Gate: exact list match, in order. **STOP GATE**: `[5,14,24,33,42]` or any shifted variant in the GGUF means the +1 was double-applied or skipped — fix before proceeding; a wrong list produces plausible-but-garbage drafts that still pass load.

- [ ] **1.9 Conv base layout golden check (risk 3, REQ-CONV-4).** Write `scripts/check_conv_base.py`: for each of the 10 conv-base tensors, assert GGUF ne == `[4096, 2, 2]` = `[n_embd, kernel, 2]`, and verify element semantics against the checkpoint: `gguf[k][t][s] == ckpt[s][t][k]` (checkpoint layout `[side, tap, channel]` — the torch→ggml reversal alone must land the right layout; reference confirms ne=[5120,2,2] with no explicit transpose). Also assert tap-0 stats dominate tap-1 (SGLang inits `base_kernel[:,0]=1.0`, dflash.py:433-435 — tap-0 mean should be O(1), tap-1 near 0). Gate: all 10 tensors pass. **STOP GATE**: layout wrong here means the whole conversion mapping is suspect.

- [ ] **1.10 block_size metadata assert (risk 5).** Assert `dflash.block_size == 8` present in the output GGUF (never absent — impl default 16 at speculative.cpp:969-975 would silently use untrained block positions). Covered by 1.7 but re-asserted standalone in `scripts/check_tensor_inventory.py`. Gate: value == 8.

- [ ] **1.11 Load sanity without server.** Two-step check that the loader accepts the file: (a) `python3 -c` snippet using the fork's `gguf-py` `GGUFReader` to print `general.architecture` (must be `dflash`) and the `dflash.*` KV set; (b) `build/bin/llama-cli -m <draft.gguf> --no-warmup -n 0 -p test 2>&1 | head -40` — the model load path runs `load_arch_hparams`/`load_arch_tensors` and prints `DFlash2 conv kernel = 2, group = 16, selector rank = 256, top-k = 16` (dflash.cpp:146) before generation even starts; CLI load alone proves no struct-level rejection. Gate: arch reads `dflash`, the dflash.cpp line appears, no `unknown architecture` / `missing tensor` / GGML_ASSERT in stderr. **STOP GATE**: loader rejection halts Sprint 1.

---

## Sprint 2 — Smoke Test (Phase B, REQ-2) — COMPLETE 2026-08-29

All gates passed: 3/3 prompts generated, draft_n > 0, zero real asserts
(`set_abort_callback` hits are benign fitting-phase lines), load log showed
`block_size=8, mask_token_id=154856, n_extract=5`, `n_max=7`. Measured:
~1.5 t/s effective, acceptance 3.36/7. See research/07-gap-analysis.md.

- [ ] **2.1 Solo-run preflight.** Write `scripts/solo_preflight.sh`: (a) verify :8086 not serving (`ss -ltn | grep :8086` empty, or `systemctl is-active llama-server-qwen-flash` ≠ active), (b) `free -g` shows ≥ 165 GB available (147 GB target + ~5 GB draft + headroom; box has 251 GB), (c) `ss -ltn | grep :8100` empty. Gate: all three pass. If production must be stopped: coordinate a window, `sudo systemctl stop llama-server-qwen-flash`, and restart it in 2.6. **STOP GATE**: never start the 147 GB load while :8086 is live.

- [ ] **2.2 Write the systemd unit.** Create `systemd/llama-server-glm5-dflash2.service` in this repo, install to `/etc/systemd/system/`. Copy the flag set from `/etc/systemd/system/llama-server-glm5.service` (port 8100, IQ4_XS 5-shard target, `--threads -1 --numa distribute --load-mode mlock`, 128k ctx, FA on, f16 KV) and add:
  ```
  --spec-type draft-dflash \
  --spec-draft-n-max 7 \
  -md /mnt/ollama/models/glm-5.3-flash/dflash2-gguf/dflash2-glm-f16.gguf
  ```
  `n_max=7` is load-bearing: `n_draft = params.n_max` and `n_block_tokens = n_draft + 1` (speculative.cpp:1236-1238) — the trained block is 8 (1 anchor + 7 drafts); default n_max would desync the block. Gate: `systemctl daemon-reload && systemctl start llama-server-glm5-dflash2` returns; unit active.

- [ ] **2.3 Load-log gates.** `journalctl -u llama-server-glm5-dflash2 -b` must show: (a) draft load line `DFlash2 conv kernel = 2, group = 16, selector rank = 256, top-k = 16` (dflash.cpp:146), (b) spec init line `block_size=8, mask_token_id=154856, n_extract=5` (speculative.cpp:999-1000), (c) target model loaded, server listening on :8100. Gate: all three lines present, values exact. **STOP GATE**: `block_size=16` in the log = metadata lost → back to 1.7; `n_extract=3` or an Eagle3-style `!= 3` throw = wrong spec type, not a fork bug (DFlash impl asserts `> 0`, spec fact 2).

- [ ] **2.4 Short generation + draft_n > 0.** With preflight from 2.1: `curl -s http://127.0.0.1:8100/v1/chat/completions -d '{"messages":[{"role":"user","content":"Write a haiku about speculative decoding."}],"max_tokens":64}'`. Parse the response `timings`: require `draft_n > 0` and `draft_n_accepted > 0` (server-common.cpp:82-85 only emits them when drafting is active). Expected: completion arrives; draft_n ≈ 7 × steps. Gate: both counters present and > 0. **STOP GATE**: draft_n == 0 or absent → drafting silently off; check `-md` load errors in journal before touching the fork.

- [ ] **2.5 No-assert sweep.** Run 3 varied prompts (short chat, ~2k-token prompt, tool-schema chat via `--jinja` endpoint). `journalctl -u llama-server-glm5-dflash2 | grep -iE "assert|GGML_ASSERT|runtime_error|abort"` must be empty. Gate: clean journal, 3/3 completions HTTP 200. **STOP GATE**: any assert → capture full backtrace, halt Sprint 2.

- [ ] **2.6 Teardown / restore.** `systemctl stop llama-server-glm5-dflash2`; if 2.1 stopped production: `sudo systemctl start llama-server-qwen-flash` and verify :8086 answers. Record smoke result (draft_n, draft_n_accepted, wall t/s from timings) in `openspec/changes/add-dflash2-support/notes.md`. Gate: production restored; smoke numbers logged.

---

## Sprint 3 — Correctness (Phase C, REQ-3)

- [ ] **3.1 Target-hidden fixture dump.** Write `tests/golden/dump_target_hiddens.cpp` (build line links `/mnt/ollama/models/llama-cpp-glm5/build/{src/libllama.a,common/libllama-common.a,ggml/src/libggml*.a}`, includes `include/` + `ggml/include/`): load IQ4_XS target, `llama_set_embeddings_layer_inp(ctx, {6,15,25,34,43}−1 …)` on a canned ~40-token prompt (agentic-style: tool-call JSON), one prefill decode, dump the 5 per-layer hidden vectors (post `build_hc_mean` collapse) + token ids to `tests/golden/fixtures/hiddens.npz`. This is the deployment-realistic hidden source (IQ4_XS target, llama.cpp extraction) — the draft is validated against exactly what it will see in production. Gate: fixture written, shapes [n_tokens, 4096] × 5. Run solo per 2.1 discipline.

- [ ] **3.2 mHC-collapse micro-test (risk 4, no GPU needed).** Write `tests/golden/test_hc_collapse.py`: replicate SGLang's mHC capture contraction (PR #36708 "hc_post + stream-average contraction", follow-up fix #36755) in pure torch against glm5next's `build_hc_mean` semantics (glm5next.cpp:613) on synthetic multi-stream tensors; assert equality within fp32 eps. Gate: pass. **STOP GATE**: mismatch means llama.cpp's collapsed hiddens differ from what the draft was distilled on — the 3.4 golden will fail downstream; resolve the reduction first (hook SGLang's exact contraction per spec risk 4).

- [ ] **3.3 SGLang reference draft dump.** Write `tests/golden/sglang_ref_dump.py`, run with `/mnt/ollama/models/glm-5.3-flash/sglang-venv/bin/python`: instantiate the DFlash2 modules from `sglang/srt/models/dflash.py` (`DFlashDraftModel` + `CandidateSelector`, BF16 weights from the local checkpoint), feed fixture hiddens through `project_target_hidden` (fc → hidden_norm), build the mask block (`[bonus, mask×7]`, positions per research/02 §3), run the draft forward + `_project_candidate_logits` (target lm_head borrow — slice the checkpoint-free path: use a random lm_head, it cancels in comparison if both sides use the same), `_score_edges` + `_follow_maps` walk, dump to `fixtures/sglang_golden.npz`: post-fc hidden, per-layer draft outputs, unary logits, lattice scores, final 7 proposed token ids. Gate: NPZ written; proposed path is 7 tokens.

- [ ] **3.4 llama.cpp replay harness.** Write `tests/golden/replay_dflash2.cpp`: load `dflash2-glm-f16.gguf`, feed the same fixture hiddens through the encoder path (`LLM_GRAPH_TYPE_ENCODER` → `batch_inject` KV materialization), run the noise block decode + selector walk (speculative.cpp:1265-1285 logic), dump the same intermediates (use the draft context's embedding-nextn output for the lattice) to `fixtures/llamacpp_replay.npz`. Gate: harness runs to completion offline (no 147 GB load needed — draft-only, ~5 GB).

- [ ] **3.5 Golden comparison.** Write `tests/golden/compare_golden.py`: elementwise rel-err `|a−b|/(|b|+1e-9)` per array; gate: draft hiddens and proposed token path match within 1e-3 rel (REQ-SD-2). Also compare the 7 proposed token ids for exact equality under the shared lm_head. **STOP GATE**: failure here catches D3 (layer off-by-one), D4 (conv layout), or mHC collapse mismatch before they masquerade as "low acceptance" — do not proceed to 3.6 with a failing golden.

- [ ] **3.6 Acceptance-length gate.** Write `scripts/bench_acceptance.py`: ~50 agentic prompts (tool-calling traces, multi-turn, JSON-heavy — store as `benchmarks/tasks/acceptance.jsonl`), POST to :8100 sequentially (concurrency 1, temperature 1.0/top-p 0.95 per the model card), accumulate `timings.draft_n` and `timings.draft_n_accepted` per request; acceptance length = `draft_n_accepted/n_steps + 1` with `n_steps = draft_n/7` (published metric counts the verifier's bonus token; published ref 5.78). Solo per 2.1. Gate: mean ≥ 5.0 over the suite (REQ-SD-3). **STOP GATE**: < 5.0 → halt; investigate in order: 1.8 layer ids, 3.5 golden, then sampling params.

- [ ] **3.7 Greedy lossless check.** Write `scripts/bench_greedy_lossless.py`: 10 prompts × `temperature=0, seed=42, max_tokens=128` against :8100 twice — spec on (draft-dflash) vs spec off (restart unit without spec flags, or second unit on :8101 run solo-serially). Compare output token id arrays for exact equality (lossless rejection sampling, REQ-SD-4). Gate: 10/10 identical. **STOP GATE**: any divergence is a correctness bug, not a tolerance question.

---

## Sprint 4 — Benchmark + Publish (Phase D, REQ-4)

- [ ] **4.1 Define the standard 3-task agentic suite.** Create `benchmarks/tasks/{toolcall,multiturn,summarize}.jsonl` — 10 prompts each: (1) single-turn tool-calling (JSON tool schemas + forced tool_choice), (2) multi-turn tool loop (3-5 rounds), (3) long-prompt (~4-8k ctx) short-output summarization. Runner `scripts/bench_agentic.py`: per task, concurrency 1, fixed seeds, records completion tokens, wall-clock, `timings.draft_n/accepted`, t/s. Gate: suite files + runner committed.

- [ ] **4.2 Baseline arm (spec off).** Solo window per 2.1: run `llama-server-glm5-dflash2` with spec flags removed (or the plain glm5 config) on :8100; run `scripts/bench_agentic.py`; record t/s per task. Cross-check against the historical 1.32 t/s (memory: glm-5-3-flash-cpu-candidate) — if the re-measured baseline deviates > ~20%, note thermals/governor (`cpu-governor.service`) before comparing arms. Gate: baseline numbers recorded.

- [ ] **4.3 Spec arm (draft-dflash on).** Same window, same suite, spec flags on (`--spec-type draft-dflash --spec-draft-n-max 7`). Record t/s, acceptance % (`draft_n_accepted/draft_n`), acceptance length, wall-clock per task. Gate: both arms measured in the same session, no :8086 overlap. **STOP GATE**: spec t/s ≤ baseline t/s → do not publish numbers as a win; diagnose (draft decode cost vs verification savings; prior art: MTP/draft-simple were net losses on Qwen CPU — bandwidth-bound boxes can lose to drafting overhead).

- [ ] **4.4 Results write-up.** `benchmarks/results-dflash2-glm.md`: table of both arms × 3 tasks (t/s, acceptance %, accept len, wall-clock), delta vs 1.32 baseline, projection check vs published 5.78/2.42×; note n_max=7, block_size=8, F16 draft, commit f30bed8. Gate: file committed with raw JSON dumps in `benchmarks/raw/`.

- [ ] **4.5 HF publish.** Create HF repo `<user>/GLM-5.3-Flash-DFlash2-GGUF` via `hf` CLI (token present at `~/.cache/huggingface/token`). Upload `dflash2-glm-f16.gguf` (+ optional Q8_0 variant via second `--outtype q8_0` conversion). README must carry: `license: cc-by-nc-nd-4.0` (weights license, risk 6 — converter code itself is MIT in this repo), `base_model: incoai/GLM-5.3-Flash-DFlash2`, inco.ai citation block (from the checkpoint README), and the serving recipe: `llama-server -m <GLM-5.3-Flash GGUF> -md <this repo> --spec-type draft-dflash --spec-draft-n-max 7`. Gate: upload lists both license tag and base_model; serving command verified verbatim against what Sprint 2 actually ran.

- [ ] **4.6 Notes to incoai + llama.cpp.** (a) z-lab/dflash GitHub discussion or issue: GLM-5.3-Flash draft converted via the merged PR #27342 converter path, CPU results table, pointer to the HF repo. (b) llama.cpp discussion (or ggml-org/llama.cpp discussions): confirmation that the merged DFlash2 path works for a glm5next target on CPU, metadata parity notes, n_max=7 gotcha, link. Gate: both posted, links recorded in `notes.md`.

- [ ] **4.7 Close out.** Update repo `README.md` status checklist (Phases 0/1/3/4 done — converter used is the fork's, note the rev-2 pivot), append findings to user memory (`glm-5-3-flash-cpu-candidate.md`: measured t/s + acceptance, replacing the 1.32-only picture). Mark OpenSpec change complete; delete `notes.md` scratch or fold into results.

---

## Sprint 5 — Gap Closure (+60% throughput, from research/07-gap-analysis.md)

Baseline for this sprint: Sprint 2 measured ~1.5 t/s (+14%), acceptance 3.36/7,
verify cost 2.2 s vs 0.87 s expected. Root causes: RC-1 MoE verify-cost
blowup, RC-2 mHC collapse mismatch + sampling mismatch. Target: ≥2.1 t/s.

- [ ] **5.1 Overhead calibration (no acceptance noise).** Solo run with
  `--spec-synth-rates 1,1,1,1,1,1,1` (forces full acceptance) and
  `--spec-synth-rates 1,0,0,0,0,0,0` (forces acceptance 2/step) to measure pure
  cycle cost at each draft length. Confirms the verify-cost model (RC-1) in one
  session. Gate: measured cycle times recorded in `benchmarks/raw/`.

- [ ] **5.2 Config sweep (fast levers).** Solo windows, one change at a time,
  `scripts/smoke_gen.sh` + a 10-prompt mini-suite for each: (a) n_max 4,
  (b) n_max 5, (c) p_min 0.4 @ n_max 7, (d) top-k 20 (all else current),
  (e) `--threads 20`, (f) `--poll 100`. Record t/s + accepted/step for each.
  Gate: best combo identified; expected ~+40% from (a)-(d). **STOP GATE** if any
  variant is slower than the Sprint-2 config — keep the sweep honest.

- [ ] **5.3 mHC extraction probe (the big fix, risk 4).** Build the Sprint 3.2
  golden test FIRST (3.1-3.5): dump SGLang reference hiddens for a canned
  prompt, replay through llama.cpp extraction, compare. Expected: mismatch —
  glm5next uses unweighted mean (models.h:1350 "collapsed by unweighted mean,
  not a gated head") vs SGLang's learned gated contraction (mhc.py:1626
  `(pre · residual).sum(1)`). Gate: quantified divergence recorded.

- [ ] **5.4 mHC fix.** If 5.3 confirms mismatch: add the gated contraction to
  the extraction path (compute `pre`/`post` mix from the model's hc_attn_fn/
  scale/base at the extraction layers, replace build_hc_mean for the dflash
  t_layer_inp path only — do NOT change normal generation). Rebuild, re-run
  golden test, then acceptance suite. Expected: acceptance 3.36 → 4.5+, t/s
  2.4–2.8. **STOP GATE**: golden test must pass at 1e-3 before any
  acceptance claims. Regression gate: run `.devgate/scripts/regression_check.py`
  after the C++ patch.

- [ ] **5.5 Re-validate losslessness.** After config + mHC changes: re-run the
  greedy spec-on == spec-off check (3.7). Gate: 10/10 identical outputs.

- [ ] **5.6 Sprint 4 benchmark with final config.** Re-run 4.2/4.3 arms with
  the winning configuration; update `benchmarks/results-dflash2-glm.md`.
  Gate: ≥ +60% over 1.32 baseline (≥2.1 t/s) or documented explanation.



- 1.2 → 1.7 (reference needed for diff). 1.5 → 1.6..1.11. Sprint 1 fully gates Sprint 2; 2.2/2.3 gate 2.4. 3.1 → 3.3/3.4 → 3.5 → 3.6 (3.2 parallel with 3.1). 3.6 + 3.7 gate Sprint 4. 4.2/4.3 share one solo window — run back-to-back to keep thermals comparable. Sprint 5: 5.1 → 5.2 → (5.3 → 5.4 → 5.5) → 5.6; golden test (5.3) is required before the mHC patch (5.4) can be claimed fixed.
- Every server task (2.x, 3.6, 3.7, 4.2, 4.3) inherits the solo-run rule: :8086 idle, preflight run, restore after. Long runs go through systemd units or `nohup … &` with log files — never a blocking foreground terminal on ucs03.
