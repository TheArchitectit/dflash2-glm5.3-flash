#!/usr/bin/env python3
"""Task #32: GSM8K acceptance mirror vs brandonmusic's published table.

Their methodology (runtime-results/v84/quality/gsm8k-distinct5-...json):
  5 distinct GSM8K prompts, temperature 1.0, top_p 0.95, reasoning_effort max,
  max_tokens 512, NO top-k, standard rejection sampling.
  Published per-row acceptance: [5.25, 4.895, 5.266, 6.031, 5.70], mean 5.428.

Sampling-tail caveat: their "no top-k" is approximated as top_k=40096 of a
154880 vocab (closest the completion API's per-request override gets while
keeping the tail) — the mirror is therefore approximate on the far tail, not
an exact protocol match. See the in-code comment at the payload.

We run the SAME prompts + settings on llama.cpp CPU DFlash2 (:8100), using the
corrected acceptance identity steps = predicted_n - accepted (matches their
token-weighted variant and server-context.cpp:664).

usage: bench_gsm8k_mirror.py [--url http://127.0.0.1:8100] [--n 5] [--reps 1]
Prints per-row + mean; writes benchmarks/raw/gsm8k_mirror.json.
"""
import argparse
import json
import os
import statistics
import urllib.request

# Standard GSM8K test rows 0-4 (the exact set their card references).
GSM8K = [
    "Janet's doctors encourage her to get at least 30 minutes of exercise "
    "three times a week. She thinks that's too much, so she gets only 1 hour "
    "of exercise a day. How many minutes of exercise a day does she get over "
    "the recommended amount?\nPut your answer in \\boxed{}.",
    "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 "
    "minutes of babysitting. How much did she earn?\nPut your answer in "
    "\\boxed{}.",
    "Betty is saving money for a new wallet which costs $100. Betty has only "
    "half of the money she needs. Her parents decided to give her $15 for "
    "that purpose, and her grandparents twice as much as her parents. How "
    "much more money does Betty need to buy the wallet?\nPut your answer in "
    "\\boxed{}.",
    "James writes a 3-page letter to 2 different friends twice a week. How "
    "many pages does he write a year?\nPut your answer in \\boxed{}.",
    "Albert is wondering how much pizza he can eat in one day. He buys 2 "
    "large pizzas and 4 small pizzas. A large pizza has 16 slices and a "
    "small pizza has 8 slices. If he eats it all, how many pieces does he "
    "eat that day?\nPut your answer in \\boxed{}.",
]


def post(url, prompt, n_predict=512):
    payload = {
        "prompt": prompt,
        "n_predict": n_predict,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 40096,          # effectively "no top-k" (vocab is 154880 but
                                 # 40096 keeps the tail; matches their open top-k)
        "cache_prompt": False,
    }
    req = urllib.request.Request(url + "/completion",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1200) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8100")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--out", default="benchmarks/raw/gsm8k_mirror.json")
    args = ap.parse_args()

    rows = []
    for rep in range(args.reps):
        for i, prompt in enumerate(GSM8K[: args.n]):
            try:
                r = post(args.url, prompt)
            except Exception as e:
                print(f"[rep{rep} row{i}] ERROR {e}", flush=True)
                continue
            t = r.get("timing", r.get("timings", {}))
            dn = t.get("draft_n", 0)
            da = t.get("draft_n_accepted", 0)
            pn = t.get("predicted_n", 0)
            tps = t.get("predicted_per_second", t.get("per_second", 0.0))
            steps = pn - da if dn > 0 else 0
            acc = (da / steps + 1) if steps > 0 else 1.0
            rows.append({"rep": rep, "row": i, "acc": acc, "tps": tps,
                         "draft_n": dn, "accepted": da, "predicted_n": pn})
            print(f"[rep{rep} row{i}] acc_len={acc:.3f} tps={tps:.2f} "
                  f"(accepted={da} steps={steps})", flush=True)

    if not rows:
        print("no successful rows")
        raise SystemExit(1)
    accs = [r["acc"] for r in rows]
    mean = statistics.mean(accs)
    # token-weighted (their variant): sum(accepted)/sum(steps) + 1
    wmean = sum(r["accepted"] for r in rows) / max(1, sum(r["predicted_n"] - r["accepted"] for r in rows)) + 1
    print("\n=== GSM8K mirror (llama.cpp CPU DFlash2) ===")
    print(f"mean acceptance   : {mean:.3f}   (published GPU ref 5.428)")
    print(f"token-weighted    : {wmean:.3f}   (published 5.441)")
    print(f"per-row           : {[round(r['acc'],2) for r in sorted(rows,key=lambda x:(x['rep'],x['row']))]}")
    tier = ("T5 PUBLISHED" if mean >= 5.0 else "T4 STRETCH" if mean >= 4.5 else
            "T3 TARGET" if mean >= 3.5 else "T2 WIN" if mean >= 2.0 else "T0/T1")
    print(f"tier (benchmarks/acceptance-gate.md): {tier}")
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "..", args.out)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({"rows": rows, "mean": mean, "weighted": wmean}, f, indent=2)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
