#!/usr/bin/env python3
"""8-prompt A/B harness (n_predict 32) with the CORRECTED acceptance identity.

steps = predicted_n - accepted  (matches server-context.cpp:664 and the
published token-weighted counting). The old /tmp/ab_bench.py used dn/7, which
inflated acc_len under p_min gating (drafts < 7) and is invalid for MTP
(n_max-dependent). Never use dn/N for step counting.

usage: ab_8prompt.py [--port 8100] [--n 8]
prints RESULT mean_acc_len=X mean_tps=Y
"""
import argparse
import json
import urllib.request

PROMPTS = [
    "Write a Python function that merges two sorted lists.",
    "Implement a binary search in Python with tests.",
    "You are an API assistant. Describe what a 204 status code means.",
    "Write a bash one-liner to find the 10 largest files under /var/log.",
    "Explain Docker multi-stage builds and why they shrink images.",
    "Write a SQL query to find duplicate email addresses in a users table.",
    "As a DevOps engineer, explain blue-green vs canary deployments.",
    "Write a Python decorator that retries a function on failure.",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--n-predict", type=int, default=32)
    args = ap.parse_args()

    acc, tps = [], []
    for p in PROMPTS[: args.n]:
        payload = {"prompt": p, "n_predict": args.n_predict, "temperature": 1.0,
                   "top_p": 0.95, "top_k": args.top_k, "min_p": 0.01,
                   "cache_prompt": False}
        req = urllib.request.Request(f"http://127.0.0.1:{args.port}/completion",
                                     data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        r = json.loads(urllib.request.urlopen(req, timeout=600).read())
        t = r["timings"]
        dn, da, pn = t.get("draft_n", 0), t.get("draft_n_accepted", 0), t.get("predicted_n", 0)
        steps = pn - da if dn > 0 else 0
        al = da / steps + 1 if steps else 1.0
        acc.append(al)
        tps.append(t.get("predicted_per_second", 0))
        print(f"  dn={dn} da={da} steps={steps} acc_len={al:.2f} "
              f"t/s={t.get('predicted_per_second', 0):.2f}", flush=True)
    print(f"RESULT mean_acc_len={sum(acc)/len(acc):.2f} mean_tps={sum(tps)/len(tps):.2f}")


if __name__ == "__main__":
    main()
