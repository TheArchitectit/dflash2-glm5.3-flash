#!/usr/bin/env bash
# gpu_ab.sh — boot llama-server with the dflash draft on GPU (any backend),
# health-gate it, run the 8-prompt A/B, tear down. Host-agnostic: point
# BIN/MODEL/DRAFT at your paths. Nothing here is measured on GPU by us
# (see docs/gpu.md "No CUDA/ROCm hardware here") — this is the harness that
# produces the first GPU numbers.
#
# usage:
#   BIN=/path/llama-server MODEL=/path/target.gguf DRAFT=/path/dflash2-glm-f16.gguf \
#   bash scripts/gpu_ab.sh --ngl 0 --ngld all --tag cpu-target-gpu-draft
#
# extra flags after -- are appended verbatim to llama-server (e.g. -- -ngl 99)
# Solo-run rule still applies on a shared box: one big model at a time; this
# script refuses to start if the port is already busy.
set -euo pipefail

BIN="${BIN:-llama-server}"
MODEL="${MODEL:?set MODEL=/path/to/GLM-5.3-Flash-shard-00001.gguf}"
DRAFT="${DRAFT:-/mnt/ollama/models/glm-5.3-flash/dflash2-gguf/dflash2-glm-f16.gguf}"
PORT="${PORT:-8100}"
TAG="gpu"
NGL="0"
NGLD="auto"
# block config: GPU users usually want the trained full block; override to
# reproduce our CPU-locked config with --n-max 4 --p-min 0.4
NMAX="${NMAX:-7}"
PMIN="${PMIN:-0.0}"
EXTRA=()

while [ $# -gt 0 ]; do
  case "$1" in
    --ngl)   NGL="$2";  shift 2 ;;
    --ngld)  NGLD="$2"; shift 2 ;;
    --tag)   TAG="$2";  shift 2 ;;
    --n-max) NMAX="$2"; shift 2 ;;
    --p-min) PMIN="$2"; shift 2 ;;
    --port)  PORT="$2"; shift 2 ;;
    --) shift; EXTRA+=("$@"); break ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

command -v "$BIN" >/dev/null 2>&1 || [ -x "$BIN" ] || { echo "BIN not found: $BIN" >&2; exit 1; }
[ -f "$MODEL" ] || { echo "MODEL not found: $MODEL" >&2; exit 1; }

# port preflight (solo-run): fail fast, don't collide with a running server
if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
  echo "port $PORT already serving — refusing to boot a second server (solo-run rule)." >&2
  echo "use a different --port, or point ab_8prompt.py at the running server directly." >&2
  exit 1
fi

LOG="/tmp/dflash2-gpu-ab-${TAG}.log"
echo "== boot: $BIN -ngl $NGL -ngld $NGLD (n_max $NMAX p_min $PMIN) -> :$PORT"
# shellcheck disable=SC2086
"$BIN" \
  -m "$MODEL" -md "$DRAFT" \
  --spec-type draft-dflash \
  --spec-draft-n-max "$NMAX" --spec-draft-p-min "$PMIN" \
  -ngl "$NGL" --spec-draft-ngl "$NGLD" \
  --host 127.0.0.1 --port "$PORT" \
  --alias unsloth/glm-5.3-flash-dflash2 \
  --ctx-size 131072 --flash-attn on \
  --cache-type-k f16 --cache-type-v f16 \
  --jinja --reasoning-effort max \
  --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.01 --repeat-penalty 1.05 \
  --no-context-shift "${EXTRA[@]}" \
  >"$LOG" 2>&1 &
SRV=$!

cleanup() {
  # defensive: kill + reap regardless of exit path; confirm the port frees
  kill "$SRV" 2>/dev/null || true
  for _ in $(seq 1 60); do
    kill -0 "$SRV" 2>/dev/null || return 0
    sleep 1
  done
  kill -9 "$SRV" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "== waiting for /health (up to 900s for a cold 147 GB load)..."
for _ in $(seq 1 900); do
  if curl -sf "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"ok"\|no_error'; then
    echo "== healthy"; break
  fi
  kill -0 "$SRV" 2>/dev/null || { echo "== server died; last 40 log lines:"; tail -40 "$LOG"; exit 1; }
  sleep 1
done
curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 || { echo "== health timeout"; tail -40 "$LOG"; exit 1; }

# confirm the offload actually happened (grep the fork's own log lines)
echo "== offload report:"
grep -E "offload|offloaded|model size|GPU[0-9] |using device|BLAS|CUDA|Vulkan|HIP|Metal" "$LOG" | head -20 || echo "  (no offload lines — CPU-only build? -ngl/-ngld were ignored)"

echo "== A/B (8-prompt, corrected acceptance ruler):"
python3 "$(dirname "$0")/ab_8prompt.py" --port "$PORT"
echo "== smoke (draft_n>0, no asserts):"
grep -cE "GGML_ASSERT|runtime_error" "$LOG" | sed 's/^/assert+err lines: /'

echo "== full log: $LOG  (append the RESULT line + your hardware to research/08)"
