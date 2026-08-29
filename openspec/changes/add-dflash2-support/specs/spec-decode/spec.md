# spec-decode — draft-dflash speculative decoding on glm5 fork

## ADDED Requirements

### Requirement: REQ-SD-1 Server smoke test

`llama-server` (glm5 fork) SHALL start with the GLM-5.3-Flash IQ4_XS target plus `--spec-type draft-dflash -md dflash2-glm.gguf`, and the smoke test SHALL pass only when the server starts, `draft_n > 0` appears in timings, and no asserts fire.

The 3-vs-5 target-layer question is already resolved (the `target_layer_ids_n != 3` assert belongs to the Eagle3 impl only; the DFlash impl asserts `> 0`) — no fork patch is expected.

#### Scenario: smoke pass

- **WHEN** `llama-server` is launched with the GLM-5.3-Flash IQ4_XS target and `--spec-type draft-dflash -md dflash2-glm.gguf`
- **THEN** the server starts, timings report `draft_n > 0`, and no asserts fire

#### Scenario: golden mismatch → halt

- **WHEN** the draft-correctness golden test fails the 1e-3 relative tolerance
- **THEN** spec-decode work is halted until the divergence is resolved (e.g. check mHC collapse `build_hc_mean` vs SGLang's extraction; hook SGLang's exact reduction if needed)

### Requirement: REQ-SD-2 Draft correctness

A golden test SHALL run the SGLang reference implementation (CPU, torch) on a canned prompt, dump draft hiddens plus the proposed path, replay the same in llama.cpp, and match within 1e-3 relative error.

Draft KV SHALL be materialized by projecting target hidden states through the draft's KV heads and writing them directly into the draft KV cache (the `batch_inject` embd path via `llama_decode` with an embd batch); unary logits SHALL borrow the target's lm_head (`_project_candidate_logits`), with the selector's top_k=16 walk (`_score_edges` + `_follow_maps`) picking the coherent 7-token path.

#### Scenario: draft hiddens match SGLang reference

- **WHEN** the canned-prompt replay in llama.cpp is compared against the SGLang reference dump
- **THEN** draft hiddens and the proposed path match within 1e-3 relative error

### Requirement: REQ-SD-3 Acceptance performance

Mean accepted tokens per step on ~50 agentic prompts SHALL be ≥ 5.0 (published reference: 5.78).

#### Scenario: acceptance below gate

- **WHEN** the mean accepted tokens over the ~50 agentic prompts is below 5.0
- **THEN** the requirement is not met and the gap is investigated before benchmark/publication phases

### Requirement: REQ-SD-4 Lossless verification

Greedy outputs with speculative decoding enabled SHALL be identical to greedy outputs with speculative decoding disabled.

#### Scenario: greedy divergence detected

- **WHEN** greedy decoding with spec on produces output differing from greedy with spec off
- **THEN** the lossless claim fails and the divergence is treated as a correctness bug
