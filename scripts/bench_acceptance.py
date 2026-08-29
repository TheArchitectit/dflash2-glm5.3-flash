#!/usr/bin/env python3
"""Sprint 3.6 — acceptance-length gate.

POSTs agentic prompts to the running speculative server (default :8100),
one at a time (concurrency 1), temp 1.0 / top-p 0.95, and accumulates
`timings.draft_n` / `timings.draft_n_accepted`. Acceptance length =
draft_n_accepted / n_steps + 1 with n_steps = draft_n / 7 (published
metric counts the verifier's bonus token; ref 5.78).

Gate (REQ-SD-3): mean acceptance-length >= 5.0.

usage: python3 scripts/bench_acceptance.py [--url http://127.0.0.1:8100]
                                           [--prompts benchmarks/tasks/acceptance.jsonl]
                                           [--n 50]
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

DEFAULT_PROMPTS = """You are a coding assistant. Write a Python function that merges two sorted lists.
Implement a binary search in Python with tests.
You are an API assistant. Describe what a 204 status code means and when to return it.
Write a bash one-liner to find the 10 largest files under /var/log.
You are a data engineer. Explain the difference between a star and snowflake schema.
Write a SQL query to find duplicate email addresses in a users table.
As a DevOps engineer, explain blue-green vs canary deployments.
Write a short JSON schema for a task object with id, title, done.
You are an agent with a get_weather tool. Call it for Paris and summarize.
Explain what a Kubernetes liveness probe does and when it matters.
Write a Python decorator that retries a function on failure.
What is the difference between TCP and UDP? Give use cases.
Write a regex that matches ISO 8601 dates.
Explain idempotency keys in REST APIs with an example.
You are a security reviewer. List three OWASP Top 10 risks and mitigations.
Write a Makefile with clean, build, and test targets.
Explain Docker multi-stage builds and why they shrink images.
Write a function to flatten a nested list recursively.
What is eventual consistency, and when is it acceptable?
Describe how a Bloom filter works and its false positives.
Write a Python context manager that times a block of code.
Explain async/await in Python with a producer-consumer example.
Write a Terraform snippet that provisions an S3 bucket with versioning.
What is a race condition? Give a Go example and a fix.
Write a git command sequence to squash the last 3 commits.
Explain the CAP theorem with a practical example.
Write a JSON object describing a user with name, email, and roles.
What is the difference between a thread and a process?
Write a Python generator that yields prime numbers.
Explain semaphores vs mutexes with code.
Write a Dockerfile for a Python FastAPI app.
What is a load balancer's role in horizontal scaling?
Write a Python function to group JSON objects by key.
Explain the SOLID principles in one line each.
Write a Node.js snippet that reads a file asynchronously.
What is the difference between 301 and 302 redirects?
Write a Python class for a circular buffer.
Explain deadlocks and how to prevent them.
Write a shell script that backs up a directory with timestamps.
What is a Git rebase vs merge? When use each?
Write a Python async function that fetches two URLs concurrently.
Explain DNS resolution steps for a domain.
Write a Python function to compute running median.
What is a connection pool and why use one?
Write a simple LRU cache in Python.
Explain TLS handshake in five steps.
Write a Python script that tails a log file.
What is the difference between precision and recall?
Write a small state machine in Python.
Comment on this code review: is defer-in-loop a bug in Go?
Explain the visitor pattern with a code sketch.""".split("\n")


def post(url, prompt, temperature, top_p):
    payload = {
        "prompt": prompt,
        "n_predict": 64,
        "temperature": temperature,
        "top_p": top_p,
        "cache_prompt": False,
    }
    req = urllib.request.Request(
        url + "/completion",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8100")
    ap.add_argument("--prompts", default=None)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    args = ap.parse_args()

    if args.prompts:
        prompts = [json.loads(l)["prompt"] for l in open(args.prompts) if l.strip()]
    else:
        prompts = [p for p in DEFAULT_PROMPTS if p.strip()]
    prompts = prompts[: args.n]

    total_accepted = 0
    total_steps = 0
    total_tps = 0.0
    n_ok = 0
    per_req = []

    for i, prompt in enumerate(prompts):
        try:
            r = post(args.url, prompt, args.temperature, args.top_p)
        except (urllib.error.URLError, json.JSONDecodeError) as e:
            print(f"[{i}] ERROR {e}", file=sys.stderr)
            continue
        t = r.get("timings", {})
        dn = t.get("draft_n", 0)
        da = t.get("draft_n_accepted", 0)
        tps = t.get("predicted_per_second", 0.0)
        steps = dn / 7 if dn > 0 else 0
        acc_len = (da / steps + 1) if steps > 0 else 1.0
        total_accepted += da
        total_steps += steps
        n_ok += 1
        per_req.append((acc_len, tps))
        if len(per_req) <= 8 or (i + 1) % 10 == 0:
            print(f"[{i+1}/{len(prompts)}] draft_n={dn} accepted={da} "
                  f"acc_len={acc_len:.2f} tps={tps:.2f}", flush=True)

    if n_ok == 0:
        print("no successful requests", file=sys.stderr)
        return 2

    mean_len = total_accepted / total_steps + 1 if total_steps else 1.0
    mean_tps = sum(p[1] for p in per_req) / len(per_req)
    print(f"\n=== acceptance-length over {n_ok} prompts ===")
    print(f"mean acceptance length : {mean_len:.3f}")
    print(f"mean t/s               : {mean_tps:.3f}")
    print(f"GATE >= 5.0            : {'PASS' if mean_len >= 5.0 else 'FAIL'}")
    return 0 if mean_len >= 5.0 else 1


if __name__ == "__main__":
    sys.exit(main())
