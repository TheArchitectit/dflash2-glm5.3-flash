#!/usr/bin/env python3
"""Sprint 4.1/4.3 runner: 3-task agentic suite against one endpoint.

Reads benchmarks/tasks/{toolcall,multiturn,summarize}.jsonl, POSTs each
prompt sequentially (concurrency 1, solo-run discipline), records wall-clock,
completion tokens, and timings.draft_n/draft_n_accepted (when spec active).

Usage:
  bench_agentic.py --port 8100 --tag spec-on [--tasks toolcall,multiturn,summarize]
Writes benchmarks/raw/{tag}_{task}.json + prints a per-task summary line:
  RESULT task=<t> prompts=<n> mean_tps=<x> mean_acc_len=<y|n/a> wall_s=<z>
"""
import argparse
import json
import os
import statistics
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
TASKS = os.path.join(HERE, "..", "benchmarks", "tasks")
RAW = os.path.join(HERE, "..", "benchmarks", "raw")


def one_request(port, rec):
    body = {"messages": rec["messages"], "max_tokens": rec["max_tokens"],
            "temperature": 0.7, "seed": 42, "cache_prompt": True}
    if rec.get("tools"):
        body["tools"] = rec["tools"]
        body["tool_choice"] = rec.get("tool_choice", "auto")
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        resp = json.loads(r.read())
    wall = time.time() - t0
    tim = resp.get("timings", {})
    out = (resp["choices"][0]["message"].get("reasoning_content") or "") + \
          (resp["choices"][0]["message"].get("content") or "")
    n = tim.get("predicted_n") or resp["usage"]["completion_tokens"]
    d, da = tim.get("draft_n"), tim.get("draft_n_accepted")
    # exact verify-step count via the server identity (server-context.cpp:664):
    # predicted_n = accepted_draft + steps -> steps = predicted_n - accepted.
    steps = (n - da) if (d and da is not None and d > 0) else 0
    acc_len = (da / steps + 1) if steps > 0 else None
    return {"id": rec["id"], "wall_s": round(wall, 2), "tokens": n,
            "tps": round(n / wall, 3), "draft_n": d, "draft_n_accepted": da,
            "acc_len": round(acc_len, 3) if acc_len else None,
            "out_len": len(out), "out": out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--tasks", default="toolcall,multiturn,summarize")
    args = ap.parse_args()
    os.makedirs(RAW, exist_ok=True)
    for task in args.tasks.split(","):
        path = os.path.join(TASKS, f"{task}.jsonl")
        recs = [json.loads(l) for l in open(path)]
        results = []
        for i, rec in enumerate(recs, 1):
            try:
                r = one_request(args.port, rec)
                results.append(r)
                print(f"[{task} {i}/{len(recs)}] tps={r['tps']} acc={r['acc_len']}",
                      flush=True)
            except Exception as e:
                print(f"[{task} {i}/{len(recs)}] ERROR {e}", flush=True)
                results.append({"id": rec["id"], "error": str(e)})
        with open(os.path.join(RAW, f"{args.tag}_{task}.json"), "w") as f:
            json.dump(results, f, indent=2)
        ok = [r for r in results if "tps" in r]
        if not ok:
            print(f"RESULT task={task} prompts=0/10 FAILED")
            sys.exit(2)
        tps = statistics.mean(r["tps"] for r in ok)
        accs = [r["acc_len"] for r in ok if r.get("acc_len")]
        acc = statistics.mean(accs) if accs else None
        wall = sum(r["wall_s"] for r in ok)
        acc_s = f"{acc:.2f}" if acc else "n/a"
        print(f"RESULT task={task} prompts={len(ok)}/{len(recs)} "
              f"mean_tps={tps:.3f} mean_acc_len={acc_s} wall_s={wall:.0f}")


if __name__ == "__main__":
    main()
