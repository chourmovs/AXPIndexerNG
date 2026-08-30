"""Portable scheduled-task daemon bootstrap independent of PYTHONPATH."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
for folder in ("shared", "daemon"):
    sys.path.insert(0, str(ROOT / folder))
os.environ.setdefault("AXPINDEXER_DATA_DIR", str(ROOT / "data"))

from axp_core.runtime import load_settings
from axp_daemon.cli import main

settings = load_settings()
if "--scheduled-task" not in sys.argv:
    raise SystemExit("This launcher is reserved for Windows Task Scheduler")
arguments = ["run", "--db", settings["db_path"], "--model-cache", settings["model_cache"],
             "--embedding-profile", settings["embedding_profile"], "--scan-interval", str(settings["scan_interval_s"]),
             "--embedding-batch-size", str(settings["embedding_batch_size"]), "--model-download-retry",
             str(settings["model_download_retry_s"]), "--launch-mode", "scheduled_task"]
if settings.get("download_missing_models", True):
    arguments.append("--allow-download")
main(arguments)
