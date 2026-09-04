#!/usr/bin/env python3
"""Validate an AXP Skill directory without loading search or AI dependencies."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "client"), str(ROOT / "shared")]

from axp_core.runtime import data_dir  # noqa: E402
from axp_client.skills import SkillStore  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Validate AXP Skill JSON files")
    parser.add_argument("--path", default=str(data_dir() / "skills"))
    args = parser.parse_args()
    store = SkillStore(args.path)
    valid, invalid = store.list_skills(), store.invalid
    print(f"{len(valid)} valid Skill{'s' if len(valid) != 1 else ''}")
    print(f"{len(invalid)} invalid Skill{'s' if len(invalid) != 1 else ''}")
    for item in invalid:
        print(f"\nINVALID {item['file']}\n  {item['detail']}")
    return 1 if invalid else 0


if __name__ == "__main__":
    raise SystemExit(main())
