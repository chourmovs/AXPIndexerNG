#!/usr/bin/env python3
"""Account for every mandatory Office file without logging document contents."""
from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "daemon"), str(ROOT / "shared")]

from axp_core.office_formats import OFFICE_EXTENSIONS  # noqa: E402
from axp_daemon.extractors import extract  # noqa: E402


def qualify(root):
    stats = defaultdict(lambda: {"seen": 0, "parsed": 0, "empty": 0, "failed": 0, "ms": 0.0})
    failures = []
    for path in sorted(Path(root).rglob("*")):
        extension = path.suffix.casefold()
        if not path.is_file() or extension not in OFFICE_EXTENSIONS:
            continue
        row = stats[extension]
        row["seen"] += 1
        started = time.perf_counter()
        try:
            sections = list(extract(path))
            if any(str(text or "").strip() for text, _ in sections):
                row["parsed"] += 1
            else:
                row["empty"] += 1
                failures.append((path, "office_content_empty"))
        except Exception as exc:  # qualification must account for the entire corpus
            row["failed"] += 1
            failures.append((path, getattr(exc, "code", type(exc).__name__)))
        row["ms"] += (time.perf_counter() - started) * 1000
    return stats, failures


def main():
    parser = argparse.ArgumentParser(description="Qualify local Office extraction coverage")
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    if not args.root.is_dir():
        parser.error(f"not a directory: {args.root}")
    stats, failures = qualify(args.root)
    print("Extension   Seen   Parsed   Empty   Failed   Mean ms")
    for extension in sorted(OFFICE_EXTENSIONS):
        row = stats[extension]
        mean = row["ms"] / row["seen"] if row["seen"] else 0.0
        print(f"{extension:<11}{row['seen']:>5}{row['parsed']:>9}{row['empty']:>8}{row['failed']:>9}{mean:>10.1f}")
    if failures:
        print("\nFailures (filename and exception type only):")
        for path, reason in failures:
            print(f"{path}: {reason}")
    accounted = all(row["seen"] == row["parsed"] + row["empty"] + row["failed"]
                    for row in stats.values())
    return 0 if accounted else 1


if __name__ == "__main__":
    raise SystemExit(main())
