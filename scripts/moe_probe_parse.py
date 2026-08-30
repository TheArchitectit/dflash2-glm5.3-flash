#!/usr/bin/env python3
"""Parse [moe-probe] log lines (LLAMA_MOE_PROBE=1, fork kernel/moe-probe)
into JSON for research/09. Lines are cumulative dumps (every 5000 calls +
atexit), so the LAST occurrence per bucket wins.

usage: moe_probe_parse.py <logfile...> [--out file.json]
prints per-bucket final stats + a derived collisions-per-verify-step note.
"""
import argparse
import json
import re

LINE = re.compile(
    r"\[moe-probe\] n(\S+) calls=(\d+) pairs=(\d+) touched=(\d+) collisions=(\d+)"
    r" col_rate=([\d.]+) cne1\[1,2,3,4-8,>8\]=(\d+),(\d+),(\d+),(\d+),(\d+)"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--out", default="-")
    args = ap.parse_args()

    final = {}
    for path in args.logs:
        with open(path, errors="replace") as f:
            for line in f:
                m = LINE.search(line)
                if not m:
                    continue
                b = m.group(1)
                final[b] = {
                    "calls": int(m.group(2)), "pairs": int(m.group(3)),
                    "touched": int(m.group(4)), "collisions": int(m.group(5)),
                    "col_rate": float(m.group(6)),
                    "cne1_hist": {"1": int(m.group(7)), "2": int(m.group(8)),
                                  "3": int(m.group(9)), "4-8": int(m.group(10)),
                                  ">8": int(m.group(11))},
                }

    # sanity per REQ-K2 review: collisions == pairs - touched (per bucket).
    # NOTE: bucket n1 can show nonzero "collisions" legitimately — a single
    # token routes to n_expert_used choices and grouped top-k can repeat an
    # expert within one token (Qwen3-Coder-Next: 16 choices -> ~9 distinct).
    # The verify-relevant signal is the COLLISION RATE DELTA vs the n1
    # baseline, not absolute collisions:
    b1 = final.get("1")
    if b1 and b1["pairs"]:
        base = b1["col_rate"]
        for b, s in final.items():
            if b in ("1", "16"):
                continue
            s["col_rate_excess_vs_n1"] = round(s["col_rate"] - base, 6)
    for b, s in final.items():
        assert s["collisions"] == s["pairs"] - s["touched"], f"bucket {b} inconsistent"

    print(json.dumps(final, indent=2))
    if args.out != "-":
        with open(args.out, "w") as f:
            json.dump(final, f, indent=2)


if __name__ == "__main__":
    main()
