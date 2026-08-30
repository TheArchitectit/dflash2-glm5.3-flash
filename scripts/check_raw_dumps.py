#!/usr/bin/env python3
"""REQ-QG5 — raw dumps are append-only evidence: detect duplicated captures.

Detects whole-content repetition (the classic double-invoked `tee -a` /
rerun-append artifact) at two granularities and k ∈ {2,3} copies:
  bytes: file equals k identical byte chunks
  lines: line list equals k identical line chunks (tolerates one trailing
         line of difference between the last copy and the rest — a final
         newline captured only in some passes)

The checker NEVER modifies dumps — evidence integrity is the point. A
flagged file is either fixed upstream (recapture) or, if the duplication
is a known accepted artifact, allowlisted with a one-line reason:

    benchmarks/raw/.duplication-allowlist
      # comments allowed; one `relativepath :: reason` per line

Exit 0 = clean or allowlisted. Exit 1 = undocumented duplication, missing
input dir, or unreadable allowlist.

usage: check_raw_dumps.py [dir]   (default: benchmarks/raw)
"""
import sys
from pathlib import Path

SUFFIXES = {".log", ".txt", ".json", ".jsonl"}
ALLOWLIST_NAME = ".duplication-allowlist"


def parse_allowlist(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    if not path.exists():
        return entries
    for ln in path.read_text().splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        if "::" not in ln:
            raise ValueError(f"allowlist line missing '::' separator: {ln!r}")
        name, reason = (part.strip() for part in ln.split("::", 1))
        if not reason:
            raise ValueError(f"allowlist entry without reason: {ln!r}")
        entries[name] = reason
    return entries


def chunked_repeat(chunks: list, k: int) -> bool:
    if len(chunks) == 0 or len(chunks) % k != 0:
        return False
    size = len(chunks) // k
    first = chunks[:size]
    return all(chunks[i * size:(i + 1) * size] == first for i in range(1, k))


def duplicated(data: bytes) -> bool:
    """True if the content is k identical copies (k in {2,3}), byte- or
    line-level, with a one-line tolerance on the final line chunk."""
    for k in (2, 3):
        if len(data) % k == 0 and chunked_repeat([data[i * len(data) // k:(i + 1) * len(data) // k] for i in range(k)], k):
            return True
    try:
        lines = data.decode("utf-8").splitlines()
    except UnicodeDecodeError:
        return False
    if len(lines) < 4:
        return False
    for k in (2, 3):
        if len(lines) % k == 0 and chunked_repeat([lines[i * len(lines) // k:(i + 1) * len(lines) // k] for i in range(k)], k):
            return True
        # one-line tolerance: the copies differ only by a trailing newline
        size, rem = divmod(len(lines), k)
        if rem and size >= 2 and chunked_repeat([lines[i * size:(i + 1) * size] for i in range(k)], k):
            return True
    return False


def main() -> int:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("benchmarks/raw")
    if not root.is_dir():
        print(f"input dir not found: {root}")
        return 1
    try:
        allow = parse_allowlist(root / ALLOWLIST_NAME)
    except ValueError as e:
        print(f"BROKEN ALLOWLIST: {e}")
        return 1

    bad: list[str] = []
    flagged_allowed = 0
    checked = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix not in SUFFIXES:
            continue
        checked += 1
        rel = path.relative_to(root).as_posix()
        if duplicated(path.read_bytes()):
            reason = allow.get(rel)
            if reason:
                flagged_allowed += 1
                print(f"ALLOWLISTED {rel} :: {reason}")
            else:
                bad.append(rel)
                print(f"DUPLICATED CONTENT: {rel} — recapture or allowlist "
                      f"with a reason in {ALLOWLIST_NAME} (never edit the dump)")

    print(f"\nchecked {checked} raw dumps: "
          f"{len(bad)} undocumented duplication(s), {flagged_allowed} allowlisted")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
