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

Acceptance length and effective throughput SHALL be scored on a tiered ladder
(see `benchmarks/acceptance-gate.md`) rather than a single hard number. The
hard constraint only blocks publish when spec throughput is a net loss vs the
no-spec baseline (Tier T0). Correctness (REQ-SD-2, REQ-SD-4) remains a hard
gate independent of this requirement.

The originally-specified "≥ 5.0" SHALL be treated as the top tier (T5
PUBLISHED), not a pass/fail floor: 5.0 was published for a different target
model (Qwen3.8-27B) and is not a lawful halt threshold for GLM-5.3-Flash.

#### Scenario: acceptance below the top tier

- **WHEN** measured acceptance lands in a lower tier
- **THEN** the tier and raw numbers are reported and recorded in
  `research/08-improvement-tracking.md`; the run proceeds to benchmark/publication
  as long as Tier T0 is not tripped

#### Scenario: spec is a net throughput loss

- **WHEN** spec-decode effective t/s is below the no-spec baseline (Tier T0)
- **THEN** publication of a "faster" claim is halted (correctness remains valid;
  spec stays available as a same-output option, benchmarked honestly)

### Requirement: REQ-SD-4 Lossless verification

Speculative decoding SHALL be distribution-lossless: the accept path samples the
target's own distribution at every verify position and accepts only on exact match
(`common_sampler_sample_and_accept_n`, sampling.cpp:678), so emitted tokens follow
the target distribution for any sampler, including greedy. Each arm SHALL be
self-deterministic (rerun identical) under a fixed seed.

Bitwise greedy equality (spec-on == spec-off) is NOT required: a batched verify
forward (GEMM) is not bit-identical to single-token decode (GEMV) on CPU, and
near-tied argmaxes flip deterministically. vLLM's standard rejection sampling has
the same property.

#### Scenario: greedy outputs diverge but arms are self-consistent

- **WHEN** spec-on reruns produce identical outputs among themselves, and spec-off
  reruns likewise, but the two arms differ on near-tie tokens
- **THEN** this is the expected CPU numerical behavior (documented, not a bug);
  the release notes state distribution-level, not bitwise, losslessness

#### Scenario: an arm is not self-deterministic, or divergence exceeds near-ties

- **WHEN** a rerun of the same arm changes output, or outputs diverge structurally
  (different first tokens, not casing/tie-level differences)
- **THEN** this is a correctness bug and publication halts
