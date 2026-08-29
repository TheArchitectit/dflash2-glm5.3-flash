#!/usr/bin/env python3
"""Sprint 3.7 greedy lossless check: spec-on == spec-off, 10/10 (REQ-SD-4).

Runs N prompts twice (temp 0, fixed seed, fixed max_tokens) against two
endpoints and compares exact token-id-equivalent output text. Per solo-run
discipline the two servers are NEVER up at the same time:
  - arm A (spec on):  :8100 (llama-server-glm5-dflash2)
  - arm B (spec off): :8101 (llama-server-glm5-nospec, run after A is stopped)

Usage: bench_greedy_lossless.py [--n 10] [--host-a :8100] [--host-b :8101]
Writes arm outputs to /tmp/lossless_{a,b}.json and compares.
"""
import argparse
import json
import sys
import urllib.request

PROMPTS = [
    "Write a Python function that merges two sorted lists in O(n).",
    "Explain the difference between TCP and UDP in 3 sentences.",
    'Call the tool `get_weather` for city "Oslo" with units metric.',
    "Summarize: Kubernetes pods are the smallest deployable units. A pod can "
    "host one or more containers that share network and storage. Deployments "
    "manage replica sets which manage pods. Services provide stable virtual IPs.",
    "Produce strict JSON: {\"name\": \"Ada\", \"year\": 1815, \"fields\": [\"math\", \"computing\"]} for Grace Hopper.",
    "Write a bash one-liner to find the 5 largest files under /var/log.",
    "What is the time complexity of quicksort worst case, and why?",
    "Draft a 3-sentence standup update: finished auth refactor, blocked on DB migration review.",
    "Translate to French: The speculative decoder drafts seven tokens and verifies them in one pass.",
    "List the first 8 prime numbers, comma-separated, nothing else.",
]


def post(host, prompt, max_tokens=128):
    body = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "seed": 42,
        "cache_prompt": True,
    }
    req = urllib.request.Request(
        f"http://{host}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=900) as r:
        resp = json.loads(r.read())
    msg = resp["choices"][0]["message"]
    # reasoning-effort=max splits output into reasoning_content + content;
    # losslessness is about the full generated stream, so compare both.
    full = (msg.get("reasoning_content") or "") + (msg.get("content") or "")
    return {"choices": [{"message": {"content": full}}],
            "usage": resp.get("usage", {}), "timings": resp.get("timings", {})}


def run_arm(host, n, tag):
    outs = []
    for i, p in enumerate(PROMPTS[:n], 1):
        try:
            resp = post(host, p)
            txt = resp["choices"][0]["message"]["content"]
            tok = resp.get("timings", {}).get("predicted_n", resp["usage"]["completion_tokens"])
            spec = resp.get("timings", {}).get("draft_n")
            outs.append({"i": i, "prompt": p, "text": txt, "tokens": tok, "draft_n": spec})
            print(f"[{i}/{n}] tokens={tok} draft_n={spec} len={len(txt)}")
        except Exception as e:
            print(f"[{i}/{n}] ERROR {e}")
            outs.append({"i": i, "prompt": p, "error": str(e)})
    with open(f"/tmp/lossless_{tag}.json", "w") as f:
        json.dump(outs, f, indent=2)
    return outs


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--host-a", default="127.0.0.1:8100")
    ap.add_argument("--host-b", default="127.0.0.1:8101")
    ap.add_argument("--arm", choices=["a", "b", "both"], default="both")
    args = ap.parse_args()

    if args.arm in ("a", "both"):
        print("=== ARM A: spec ON ===")
        a = run_arm(args.host_a, args.n, "a")
        if any("error" in o for o in a):
            print("ARM A INCOMPLETE — fix before arm B (comparison would be void)")
            sys.exit(2)
    if args.arm in ("b", "both"):
        print("=== ARM B: spec OFF ===")
        b = run_arm(args.host_b, args.n, "b")

    if args.arm == "both":
        a = json.load(open("/tmp/lossless_a.json"))
        b = json.load(open("/tmp/lossless_b.json"))
        match = 0
        for x, y in zip(a, b):
            if "error" in x or "error" in y:
                print(f"[{x['i']}] SKIP (request error)")
                continue
            ok = x["text"] == y["text"]
            match += ok
            mark = "OK " if ok else "DIFF"
            print(f"[{x['i']}] {mark} a_len={len(x['text'])} b_len={len(y['text'])}")
            if not ok:
                # first divergence point
                xa, yb = x["text"], y["text"]
                d = next((k for k in range(min(len(xa), len(yb))) if xa[k] != yb[k]), min(len(xa), len(yb)))
                print(f"     first diff at char {d}: a={xa[max(0,d-20):d+20]!r} b={yb[max(0,d-20):d+20]!r}")
        print(f"\n=== lossless: {match}/{len(a)} identical ===")
        print("GATE 10/10 :" , "PASS" if match == len(a) == 10 else "FAIL")
        sys.exit(0 if match == len(a) == 10 else 1)


if __name__ == "__main__":
    main()
