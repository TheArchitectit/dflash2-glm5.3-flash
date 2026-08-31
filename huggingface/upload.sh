#!/usr/bin/env bash
# T6 — HF publish (DEFERRED until the user says go; run manually).
# The token's HF account is `lundrog` (verified via `hf auth whoami`), not
# `user001` as an earlier draft assumed. Override REPO to change namespace.
#
# Gate checklist (from openspec T6) — all pass, nothing left to measure:
#   [x] F16 + Q8_0 + BF16 each pass check_tensor_inventory + check_conv_base
#       + diff_gguf_meta (T1/T2, research/08-improvement-tracking.md)
#   [x] serving recipe verified by curl against live :8100 (2026-08-30)
#   [x] model-card.md carries license/base_model/bench table/results pointer
#   [x] secret scan (gitleaks) clean; GitHub repo carries zero binaries
set -euo pipefail
REPO="${REPO:-lundrog/GLM-5.3-Flash-DFlash2-GGUF}"
SRC="${SRC:-/mnt/ollama/models/glm-5.3-flash/dflash2-gguf}"
cd "$(dirname "$0")/.."

echo "[1/4] upload F16 first (this creates $REPO private via --private)"
hf upload "$REPO" --repo-type model --private \
  "$SRC/dflash2-glm-f16.gguf" dflash2-glm-f16.gguf

echo "[2/4] remaining files in parallel (repo exists now)"
hf upload "$REPO" --repo-type model \
  "$SRC/dflash2-glm-q8_0.gguf" dflash2-glm-q8_0.gguf &
P1=$!
hf upload "$REPO" --repo-type model \
  "$SRC/dflash2-glm-bf16.gguf" dflash2-glm-bf16.gguf &
P2=$!
hf upload "$REPO" --repo-type model \
  huggingface/model-card.md README.md &
P3=$!
wait $P1 $P2 $P3

echo "[3/4] download-back smoke gate (REQ-T6): refetch F16 + inventory"
TMP=$(mktemp -d)
hf download "$REPO" --repo-type model dflash2-glm-f16.gguf --local-dir "$TMP"
python3 scripts/check_tensor_inventory.py "$TMP/dflash2-glm-f16.gguf"
rm -rf "$TMP"

echo "[4/4] gates passed -> flipping public"
# Making the repo public is an outward-facing, hard-to-walk-back action:
# require an explicit yes (or FLIP_PUBLIC=1 for scripted runs).
if [ "${FLIP_PUBLIC:-}" != "1" ]; then
    read -r -p "Make $REPO public now? Type 'yes' to confirm: " ans
    if [ "$ans" != "yes" ]; then
        echo "ABORTED — $REPO stays private. Re-run with FLIP_PUBLIC=1 to skip this prompt."
        exit 1
    fi
fi
python3 -c "
from huggingface_hub import HfApi
HfApi().update_repo_visibility('$REPO', private=False)
print('PUBLIC:', '$REPO')"
