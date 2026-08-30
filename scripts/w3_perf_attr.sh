#!/usr/bin/env bash
# w3_perf_attr.sh — REQ-K3 op-class attribution: perf-sampling the verify
# workload (synth full-acceptance, n_max=7) to split the b*n marginal cost
# across mul_mat_id / ssm(KDA) / attention. Attaches to the pid AFTER the
# 147GB load so samples are decode-only. Run inside a solo window.
set -uo pipefail

BIN="${BIN:-/mnt/ollama/models/llama-cpp-kernel/build-probe/bin/llama-server}"
MODEL="${MODEL:-/mnt/ollama/models/glm-5.3-flash/UD-IQ4_XS/GLM-5.3-Flash-UD-IQ4_XS-00001-of-00005.gguf}"
DRAFT="${DRAFT:-/mnt/ollama/models/glm-5.3-flash/dflash2-gguf/dflash2-glm-f16.gguf}"
PORT="${PORT:-8100}"
LOG="/tmp/dflash2-k3.log"
DATA="/tmp/dflash2-k3.data"

avail=$(free -g | awk 'NR==2{print $7}')
[ "$avail" -ge 200 ] || { echo "PREFLIGHT FAIL: ${avail}GB" >&2; exit 1; }
curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { echo "port busy" >&2; exit 1; }

"$BIN" \
  -m "$MODEL" -md "$DRAFT" \
  --spec-type draft-dflash --spec-draft-n-max 7 --spec-draft-p-min 0 \
  --spec-synth-rates 1,1,1,1,1,1,1 \
  --alias unsloth/glm-5.3-flash-dflash2 \
  --host 127.0.0.1 --port "$PORT" \
  --threads -1 --numa distribute \
  --parallel 1 --ctx-size 16384 --flash-attn on \
  --cache-type-k f16 --cache-type-v f16 \
  --batch-size 2048 --ubatch-size 512 \
  --jinja --fit off --reasoning-effort max \
  --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.01 --repeat-penalty 1.05 \
  --no-context-shift >"$LOG" 2>&1 &
SRV=$!
PERF=""
cleanup() {
  [ -n "$PERF" ] && kill -INT "$PERF" 2>/dev/null && wait "$PERF" 2>/dev/null
  kill "$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

for _ in $(seq 1 1800); do
  curl -sf "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q 'ok\|no_error' && break
  kill -0 "$SRV" 2>/dev/null || { echo "died"; tail -20 "$LOG"; exit 1; }
  sleep 1
done
echo "== healthy; attaching perf to pid $SRV"

perf record -F 99 --call-graph=lbr -o "$DATA" -p "$SRV" &
PERF=$!
sleep 3

# sustained decode workload: long generations, full-acceptance verify steps
for rep in 1 2 3 4 5 6; do
  curl -s "http://127.0.0.1:$PORT/completion" \
    -d '{"prompt":"Write a detailed technical explanation, with examples and edge cases, of how block-diffusion speculative decoding amortizes target-model weight reads on a bandwidth-bound CPU, and what limits remain.","n_predict":512,"temperature":1.0,"top_p":0.95,"top_k":20,"min_p":0.01,"cache_prompt":false}' \
    | python3 -c "import json,sys; t=json.load(sys.stdin).get('timings',{}); print(f\"RESULT rep=$rep tokrate={t.get('predicted_per_second',0):.4f} ngen={t.get('predicted_n')}\")"
done

sleep 1
kill -INT "$PERF"; wait "$PERF" 2>/dev/null; PERF=""
echo "== perf report (top symbols):"
perf report --stdio -i "$DATA" --percent-limit 0.3 2>/dev/null | grep -aE "^ +[0-9]" | head -35 > /tmp/dflash2-k3-report.txt
cat /tmp/dflash2-k3-report.txt
echo "DONE-W3"
