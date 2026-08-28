"""Portable tray bootstrap independent of system Python and PYTHONPATH."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for folder in ("shared", "daemon", "client", "tray"):
    sys.path.insert(0, str(ROOT / folder))
os.environ.setdefault("AXPINDEXER_DATA_DIR", str(ROOT / "data"))

from axp_tray.app import main

main()
