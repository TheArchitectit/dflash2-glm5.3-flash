#!/usr/bin/env bash
# k1_cost_curve.sh — verify cost curve point via forced full acceptance
# (openspec: add-moe-kernel-tuning, REQ-K1). Boots the probe build at a
# given n_max with synth rates = all-1 (rates.size must == n_max — the fork
# asserts it), runs N reps of a fixed prompt, parses server-side decode
# telemetry, kills the server. ONE point per invocation (one cold load is
# the cost; loop over configs outside).
#
# usage: K1_NMAX=7 bash scripts/k1_cost_curve.sh            # cost(8-token verify)
#        K1_NMAX=4 bash scripts/k1_cost_curve.sh            # cost(5-token verify)
#        LLAMA_MOE_PROBE=1 K1_NMAX=7 ... # W2 rides probe in the same window
#
# solo-run: preflight RAM>=200, port free; restores nothing (user: no
# production on this host) but never leaves its own server running.
set -uo pipefail

BIN="${BIN:-/mnt/ollama/models/llama-cpp-kernel/build-probe/bin/llama-server}"
MODEL="${MODEL:-/mnt/ollama/models/glm-5.3-flash/UD-IQ4_XS/GLM-5.3-Flash-UD-IQ4_XS-00001-of-00005.gguf}"
DRAFT="${DRAFT:-/mnt/ollama/models/glm-5.3-flash/dflash2-gguf/dflash2-glm-f16.gguf}"
PORT="${PORT:-8100}"
NMAX="${K1_NMAX:?set K1_NMAX (7 or 4)}"
REPS="${REPS:-3}"
NPRED="${NPRED:-64}"
TAG="${TAG:-k1_nmax${NMAX}}"
LOG="/tmp/dflash2-k1-${TAG}.log"

# rates: NMAX x "1,"
RATES=$(python3 -c "print(','.join(['1']*$NMAX))")

avail=$(free -g | awk 'NR==2{print $7}')
[ "$avail" -ge 200 ] || { echo "PREFLIGHT FAIL: ${avail}GB < 200GB — stop the other server first" >&2; exit 1; }
if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
  echo "port $PORT busy — solo-run conflict, refusing" >&2; exit 1
fi

"$BIN" \
  -m "$MODEL" -md "$DRAFT" \
  --spec-type draft-dflash --spec-draft-n-max "$NMAX" --spec-draft-p-min 0 \
  --spec-synth-rates "$RATES" \
  --alias unsloth/glm-5.3-flash-dflash2 \
  --host 127.0.0.1 --port "$PORT" \
  --threads -1 --numa distribute --load-mode mlock \
  --parallel 1 --ctx-size 16384 --flash-attn on \
  --cache-type-k f16 --cache-type-v f16 \
  --batch-size 2048 --ubatch-size 512 \
  --jinja --fit off --reasoning-effort max \
  --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.01 --repeat-penalty 1.05 \
  --no-context-shift --verbose \
  >"$LOG" 2>&1 &
SRV=$!
cleanup() { kill "$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

echo "== waiting for /health (cold 147GB load; up to 1800s)..."
for _ in $(seq 1 1800); do
  curl -sf "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q 'ok\|no_error' && { echo "== healthy"; break; }
  kill -0 "$SRV" 2>/dev/null || { echo "== died; log tail:"; tail -30 "$LOG"; exit 1; }
  sleep 1
done

for i in $(seq 1 "$REPS"); do
  echo "-- rep $i"
  curl -s "http://127.0.0.1:$PORT/completion" \
    -d "{\"prompt\":\"Explain in detail how speculative decoding amortizes weight reads on a bandwidth-bound CPU, with examples.\",\"n_predict\":$NPRED,\"temperature\":1.0,\"top_p\":0.95,\"top_k\":20,\"min_p\":0.01,\"cache_prompt\":false}" \
    | python3 -c "
import json,sys
d=json.load(sys.stdin); t=d.get('timings',{})
n=$NMAX+1; rate=max(t.get('predicted_per_second',1e-9),1e-9)
# synth full-accept: each decode step emits n_max+1 tokens -> cycle_s = step wall time
print(f\"RESULT rep=$i ngen={t.get('predicted_n')} tokrate={rate:.4f} cycle_s={n/rate:.4f} ev_ms={t.get('embedding_ms',0):.1f}\")"
done

# graceful stop (server sigaction's SIGTERM -> normal exit -> atexit probe dump)
kill "$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null || true; sleep 2

grep -aE "mean accept|draft-dflash" "$LOG" | tail -3
grep -a "moe-probe" "$LOG" | tail -12
echo "DONE-K1 tag=$TAG"
