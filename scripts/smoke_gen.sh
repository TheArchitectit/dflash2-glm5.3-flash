#!/usr/bin/env bash
# Sprint 2.4/2.5 — short generation + no-assert sweep against :8100 (dflash2).
# Usage: bash smoke_gen.sh  (server must already be up)
set -u
PORT=8100
LOG=/tmp/dflash2-smoke.json
PASS=0; FAIL=0

gen() {
  local label="$1" body="$2"
  local resp
  resp=$(curl -s "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' -d "$body")
  local http_ok=$?
  local content draft_n accepted
  content=$(echo "$resp" | python3 -c "
import json,sys
d=json.load(sys.stdin)
m=d['choices'][0]['message']
print((m.get('content') or m.get('reasoning_content') or '')[:200])" 2>/dev/null)
  local t=$(echo "$resp" | python3 -c "
import json,sys
d=json.load(sys.stdin)
t=d.get('timings',{})
print(f\"draft_n={t.get('draft_n','ABSENT')} draft_n_accepted={t.get('draft_n_accepted','ABSENT')}\")" 2>/dev/null)
  if [ -n "$content" ] && [[ "$t" == *draft_n=* ]] && [[ "$t" != *ABSENT* ]]; then
    echo "PASS [$label] $t"
    echo "  output: ${content:0:120}..."
    PASS=$((PASS+1))
  else
    echo "FAIL [$label] no content or no draft timings: $t"
    echo "$resp" | head -c 400
    FAIL=$((FAIL+1))
  fi
}

echo "=== Sprint 2 smoke: 3 prompts ==="
gen "short-chat" '{"messages":[{"role":"user","content":"Write a haiku about speculative decoding."}],"max_tokens":64}'
gen "medium-prompt" '{"messages":[{"role":"user","content":"Explain in two sentences how block-diffusion language models draft multiple tokens per step, and why this helps bandwidth-bound CPUs."}],"max_tokens":128}'
gen "tool-schema" '{"messages":[{"role":"user","content":"Use the get_weather tool for Tokyo."}],"tools":[{"type":"function","function":{"name":"get_weather","description":"Get weather","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}],"max_tokens":128}'

echo "=== assert sweep ==="
HITS=$(journalctl -u llama-server-glm5-dflash2 -b --no-pager | grep -E "GGML_ASSERT|runtime_error|abort\b" | grep -v "set_abort_callback" | wc -l)
echo "journal assert/error hits: $HITS"
echo "=== RESULT: $PASS passed, $FAIL failed, $HITS asserts ==="
[ "$FAIL" -eq 0 ] && [ "$HITS" -eq 0 ]
