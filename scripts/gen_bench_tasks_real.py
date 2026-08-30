#!/usr/bin/env python3
"""Real-repo benchmark suite v2 — prompts mined from /mnt/ollama/git.

The synthetic suite (gen_bench_tasks.py) used invented prompts. This one uses
the user's real working repos, which is what agentic DFlash2 acceptance should
actually be measured on:

  toolcall   -> real imperative commit messages as instructions, real source
               files as tool results
  multiturn  -> commit message -> real diff -> follow-up (3 round trips)
  summarize  -> concatenated real README/source, 4-8k tokens per prompt

Safety for a repo that may go public:
  * generated *_v2.jsonl files are NOT committed (.gitignore) - only this
    generator ships; tasks regenerate locally from the repos.
  * repos whose name contains "private" are excluded.
  * every extracted snippet passes the same secret regex battery as the
    release scrub; any hit drops the snippet (and is reported).
Provenance: each record carries source = {repo, path|commit}.

usage: gen_bench_tasks_real.py [--root /mnt/ollama/git] [--quiet]
"""
import argparse
import json
import os
import re
import subprocess

SECRET_PATTERNS = [
    r"hf_[A-Za-z0-9]{20,}",
    r"sk-[A-Za-z0-9]{20,}",
    r"ghp_[A-Za-z0-9]{20,}",
    r"github_pat_[A-Za-z0-9_]{20,}",
    r"AKIA[0-9A-Z]{16}",
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
    r"xox[baprs]-[A-Za-z0-9-]{10,}",
    r"AIza[0-9A-Za-z_-]{30,}",
    r"(?i:password)\s*[=:]\s*['\"][^'\"]{6,}",
]
SECRET_RE = re.compile("|".join(SECRET_PATTERNS))

# repos to mine (python-ish, agentic workloads; no *-private)
REPOS = {
    "claw-code": ["claw", "tests"],
    "llama.cpp-repo": ["examples", "gguf-py"],
    "radvulscanner": ["src", "radvulscanner"],
    "agent-guardrails-template": ["scripts", "src"],
}


def g(cwd, *args):
    try:
        return subprocess.run(["git", "-C", cwd, *args], capture_output=True,
                              text=True, timeout=30).stdout
    except Exception:
        return ""


def clean(text, limit=1200):
    text = text[:limit]
    if SECRET_RE.search(text):
        return None
    return text.strip()


def commit_messages(repo_dir, n):
    out = g(repo_dir, "log", "--format=%s", "-n", str(n * 3))
    msgs = [m for m in out.splitlines() if 15 < len(m) < 120 and clean(m, 9999)]
    return msgs[:n]


def pick_files(repo_dir, subdirs, exts, count):
    files = []
    for sd in subdirs:
        root = os.path.join(repo_dir, sd)
        for dp, _, fns in os.walk(root):
            if ".git" in dp or "node_modules" in dp:
                continue
            for fn in fns:
                if any(fn.endswith(e) for e in exts):
                    files.append(os.path.join(dp, fn))
    files.sort(key=lambda p: os.path.getsize(p))
    mid = files[len(files)//4 : len(files)]
    step = max(1, len(mid) // count)
    return mid[::step][:count]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/mnt/ollama/git")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    out_dir = os.path.join(os.path.dirname(__file__), "..", "benchmarks", "tasks")
    os.makedirs(out_dir, exist_ok=True)
    dropped = 0

    def emit(name, rows):
        path = os.path.join(out_dir, name)
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"wrote {path} ({len(rows)})")

    # ---- toolcall: commit message as the instruction, real file as tool result
    tc, mt, sm = [], [], []
    for repo, subdirs in REPOS.items():
        d = os.path.join(args.root, repo)
        if "private" in repo.lower() or not os.path.isdir(d):
            continue
        msgs = commit_messages(d, 12)
        files = pick_files(d, subdirs, (".py", ".ts", ".js"), 12)
        per = max(1, len(msgs) // max(1, len(files)))
        for j, fp in enumerate(files[:10]):
            content = open(fp, errors="ignore").read(1400)
            content = clean(content)
            if not content:
                dropped += 1
                continue
            msg = msgs[j * per] if j * per < len(msgs) else msgs[0]
            rel = os.path.relpath(fp, d)
            tc.append({"id": f"v2-toolcall-{len(tc):02d}", "task": "toolcall",
                       "kind": "real_repo_toolcall",
                       "source": {"repo": repo, "path": rel},
                       "messages": [{"role": "user",
                                     "content": f"As the repo's coding agent, "
                                     f"implement this change: {msg}\nFirst show "
                                     f"the current {rel} you are modifying."}],
                       "tools": [{"type": "function", "function": {
                           "name": "read_file",
                           "parameters": {"type": "object",
                                          "properties": {"path": {"type": "string"}},
                                          "required": ["path"]}}}],
                       "tool_choice": "required", "max_tokens": 256})
        # multiturn: message -> diff -> follow-up
        for h in g(d, "log", "--format=%H", "-n", "6").splitlines()[:3]:
            body = clean(g(d, "show", "--stat", "--format=%s%n%b", "-s", h) +
                         g(d, "show", "--format=", "-U1", h)[:1200], 1600)
            if not body:
                dropped += 1
                continue
            idx = len(mt)
            mt.append({"id": f"v2-multiturn-{idx:02d}", "task": "multiturn",
                       "kind": "real_repo_diff_review",
                       "source": {"repo": repo, "commit": h[:10]},
                       "messages": [
                           {"role": "user", "content":
                            f"Review this commit message and diff from {repo} "
                            f"for bugs and style issues:\n\n{body[:1200]}"},
                           {"role": "assistant", "content": "Reading the diff."},
                           {"role": "tool", "name": "read_file",
                            "content": body[1200:2400] or body[:1000]},
                           {"role": "user", "content":
                            "Give the top 3 concrete fixes, each with the file "
                            "and a one-line patch description."}],
                       "max_tokens": 256})
        # summarize: real source concatenated to 4-8k tokens
        blob_parts = []
        for fp in files:
            blob_parts.append(f"### {os.path.relpath(fp, d)}\n" +
                              open(fp, errors="ignore").read(2000))
        blob = clean("\n\n".join(blob_parts), 20000)
        if not blob:
            dropped += 1
            continue
        idx = len(sm)
        sm.append({"id": f"v2-summarize-{idx:02d}", "task": "summarize",
                   "kind": "real_repo_code_summarize",
                   "source": {"repo": repo, "files": len(files)},
                   "messages": [{"role": "user", "content":
                                 "Summarize what this code does in 5 bullets, "
                                 "then list its key data structures:\n\n" + blob}],
                   "max_tokens": 192})

    # top up to 10 each by cycling (keeps >=10 rows even if a repo is small)
    def topup(rows, n=10):
        if not rows:
            return rows
        out = list(rows)
        i = 0
        while len(out) < n:
            r = dict(rows[i % len(rows)])
            r["id"] = f"{r['id']}r{len(out)}"
            out.append(r)
            i += 1
        return out[:n]

    emit("toolcall_v2.jsonl", topup(tc))
    emit("multiturn_v2.jsonl", topup(mt))
    emit("summarize_v2.jsonl", topup(sm))
    if dropped and not args.quiet:
        print(f"DROPPED {dropped} snippets failing secret scan")


if __name__ == "__main__":
    main()
