# Proposal: release-v001-dynamic — close out v0.0.1 as an autonomous loop

## Why

The remaining release work (4.5 variants → secret scan → public repo → HF
publish → notes → tag) is a serial queue of long-running, mostly-unattended
steps (bench arms take 30–90 min each). We already run it as a dynamic
/loop with persistent monitors; this change formalizes that protocol so it
survives compaction/restart and so the rules are reviewable, not tribal.

## What Changes

- Formalize the **dynamic workflow contract** (design.md): wake signals,
  watchdogs, solo-run discipline, commit-per-step, escalation triggers.
- Task list for the queue tail: 4.5 A/B, close-out chain (scan → public →
  HF), 4.7/4.8.
- No code changes to llama.cpp or the draft path.

## Impact

- AF affects: benchmarks/, research/08-improvement-tracking.md, this repo's
  public visibility, new HF repo user001/GLM-5.3-Flash-DFlash2-GGUF.
- OUT: Qwen production :8086 (never touched — standing user rule).
- OUT: the glm5 fork (no patches needed for release).
