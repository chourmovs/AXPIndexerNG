#!/usr/bin/env python3
"""Run the same local model qualification engine used by Manage Local AI."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for package_root in (ROOT / "client", ROOT / "shared"):
    sys.path.insert(0, str(package_root))

from axp_client.rag.model_catalog import catalog_model  # noqa: E402
from axp_client.rag.model_manager import ModelManager  # noqa: E402
from axp_client.rag.runtime_manager import InferenceRuntimeManager  # noqa: E402
from axp_core.runtime import load_settings  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Qualify installed AXP catalog chat models locally")
    parser.add_argument("--profile", choices=("standard", "stress"), default="standard")
    parser.add_argument("--model", help="qualify one catalog model id")
    parser.add_argument("--cache", type=Path, help="model cache root (defaults to configured cache)")
    args = parser.parse_args()
    settings = load_settings()
    if args.model and catalog_model(args.model) is None: parser.error("unknown catalog model")
    cache = args.cache or Path(settings.get("model_cache", ROOT / "model-cache"))
    runtime = InferenceRuntimeManager(settings)
    manager = ModelManager(cache, runtime=runtime)
    try:
        manager.start_qualification(args.profile, args.model)
        runner = manager._qualification
        while runner.job.state not in ("complete", "complete_with_errors", "failed", "cancelled"):
            __import__("time").sleep(.2)
        print(json.dumps(runner.job.report, indent=2))
        return 0 if runner.job.state in ("complete", "complete_with_errors") else 1
    except KeyboardInterrupt:
        manager.cancel_qualification()
        return 130
    finally:
        runtime.close()


if __name__ == "__main__":
    raise SystemExit(main())
