# Release Plan: GLM-5.3-Flash-DFlash2-GGUF on Hugging Face

Single source of truth for getting from "working smoke test" to "published,
verified, near-perfect release." Owner: user001. Box: ucs03 (CPU-only).

## Definition of done (the release is DONE when all are true)

1. **Artifact**: F16 GGUF (+Q8_0 if it passes the acceptance A/B) on
   `user001/GLM-5.3-Flash-DFlash2-GGUF`, passing all three Sprint-1 gate
   scripts (tensor inventory, metadata parity vs incoai reference, conv-base
   byte check).
2. **Correctness**: golden test vs SGLang reference within 1e-3 rel; greedy
   lossless check 10/10 identical (spec on == spec off).
3. **Performance**: benchmark table (baseline vs spec, 3-task agentic suite,
   solo runs); spec arm > baseline, target ≥ +60% over the 1.32 t/s baseline.
4. **Model card**: license (cc-by-nc-nd-4.0), base_model, inco.ai citation,
   benchmark table, exact serving recipe (curl-verified), n_max/block_size
   gotchas, the mHC finding if still open.
5. **Gates**: DevGate guardrails/regression/tests green; all repo files < 500
   lines; repo tagged; notes posted to z-lab/dflash + llama.cpp discussions.

## The path (sprints, in order)

| Sprint | Contents | Status |
|---|---|---|
| 1 | Conversion + parity gates | ✅ done 2026-08-29 |
| 2 | Smoke test (draft_n>0, no asserts) | ✅ done 2026-08-29 |
| 3 | Golden test (mHC divergence quantified) + acceptance ≥5.0 + lossless 10/10 | ⬜ **publish blocker** |
| 4 | Bench suite, results write-up, quant variants, HF upload, notes, tag | ⬜ |
| 5 | Gap closure: config sweep + mHC fix + re-validation + final bench | ⬜ |

Sprint 3 and Sprint 5 interleave: run 5.1–5.2 (config) first (fast wins,
informative), then 3.1–3.5 (golden) which gates the 5.4 mHC fix, then 5.5
re-validation, then Sprint 4 in full.

## Known open items (from research/07-gap-analysis.md)

- mHC collapse mismatch CONFIRMED (risk 7): fix is Sprint 5.4, gated by the
  3.5 golden test. Until fixed, acceptance sits ~3.4 vs published 4.4–5.5 —
  publishable with honest numbers, but the fix is worth doing first.
- MoE verify-cost blowup (risk 8): config levers (n_max, p_min) recover ~+40%;
  kernel-level fixes out of scope for v1.
- Sampling: publish the config actually benchmarked (top-k, p_min, n_max) so
  users can reproduce our numbers.

## Solo-run discipline (applies to every server task)

:8086 stopped → preflight (ports + RAM) → run via systemd + logs → restore
Qwen after. Never concurrent, never blocking foreground runs.

## HF upload checklist (4.6, in order)

1. `hf repo create user001/GLM-5.3-Flash-DFlash2-GGUF --type model`
2. Upload F16 GGUF (+Q8_0) — large-file upload, use `hf upload`
3. Write README.md (model card) with: front-matter license/base_model, inco.ai
   citation, benchmark table, serving recipe, gotchas
4. Verify: `curl` the repo URL → 200; `hf download` round-trip →
   `check_tensor_inventory.py` passes on the downloaded file
5. Post notes to z-lab/dflash + llama.cpp discussions
6. Tag repo `v1.0.0-dflash2-glm`
