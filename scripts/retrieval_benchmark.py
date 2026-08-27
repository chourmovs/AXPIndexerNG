"""Model-enabled benchmark entry point; intentionally opt-in, never part of normal CI."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()
    if not args.allow_download:
        raise SystemExit("Real-model benchmark requires explicit --allow-download")
    cases = json.loads((Path(__file__).parents[1] / "tests/retrieval_cases/golden.json").read_text())
    print(f"Provisioning/model-enabled harness ready: {len(cases)} golden cases")
    print("Run the indexed corpus through fts, dense, hybrid, and quality profiles to print Recall@5 and MRR@10.")


if __name__ == "__main__":
    main()
