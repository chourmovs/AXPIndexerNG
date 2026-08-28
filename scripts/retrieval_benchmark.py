"""Opt-in real-model quality retrieval smoke test."""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).parents[1]
for folder in ("shared", "daemon", "client"):
    sys.path.insert(0, str(ROOT / folder))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()
    if not args.allow_download:
        raise SystemExit("Real-model benchmark requires explicit --allow-download")

    from axp_client.reranker import Reranker
    from axp_daemon.embeddings import Embedder

    with tempfile.TemporaryDirectory() as temporary:
        work = Path(temporary)
        cache = work / "model-cache"
        documents = work / "documents"
        documents.mkdir()
        (documents / "reactor.txt").write_text("R042500 reactor overpressure relief procedure.", encoding="utf-8")
        db = work / "quality-smoke.db"

        # Explicit provisioning is confined to this optional workflow.
        Embedder("balanced", cache_dir=cache, local_only=False)
        Reranker(cache_dir=cache, local_only=False)
        env = os.environ | {
            "FASTEMBED_CACHE_PATH": str(cache),
            "PYTHONPATH": os.pathsep.join(str(ROOT / part) for part in ("shared", "daemon", "client")),
        }
        subprocess.run(
            [
                sys.executable,
                "-m",
                "axp_daemon",
                "scan",
                "--db",
                str(db),
                "--root",
                str(documents),
            ],
            check=True,
            env=env,
        )
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "axp_client",
                "search",
                "--db",
                str(db),
                "--query",
                "R042500 reactor pressure",
                "--profile",
                "quality",
                "--explain",
            ],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )
        response = json.loads(completed.stdout)
        first = response["results"][0]
        assert first["reranker_score"] is not None
        assert first["rerank_rank"] is not None
        print(json.dumps({"quality_reranker_smoke": "passed", "result": first}, ensure_ascii=False))


if __name__ == "__main__":
    main()
