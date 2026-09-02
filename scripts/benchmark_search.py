#!/usr/bin/env python3
"""Report cold/warm Search timings against an existing, unmodified index."""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT, ROOT / "shared", ROOT / "daemon", ROOT / "client"):
    sys.path.insert(0, str(folder))

from axp_client.search import search  # noqa: E402
from axp_core.database import SearchReaderPool, connect  # noqa: E402
from axp_daemon.embeddings import embedder_for_index  # noqa: E402

QUERIES = (
    "n-Heptane 99% density",
    "density of MTBE",
    "Heptane",
    "ammoniaque",
    "MSDS MTBE SIMFEX",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, type=Path)
    args = parser.parse_args()
    with connect(args.db, readonly=True) as con:
        embedder = embedder_for_index(con)
    for query in QUERIES:
        pool = SearchReaderPool(args.db, size=2)
        try:
            for temperature in ("cold", "warm"):
                acquire_started = time.perf_counter()
                with pool.acquire() as (con, reused):
                    db_acquire_ms = (time.perf_counter() - acquire_started) * 1000
                    result = search(con, embedder, query, explain=True)
                    timing = result["timings"]
                    row = {
                        "query": query,
                        "cold/warm": temperature,
                        "reader_reused": reused,
                        "db_acquire_ms": db_acquire_ms,
                        "fts_ms": timing.get("fts_retrieval_ms", 0.0),
                        "vector_ms": timing.get("vector_retrieval_ms", 0.0),
                        "fusion_ms": timing.get("fusion_ms", 0.0),
                        "scoring_ms": timing.get("passage_scoring_ms", 0.0),
                        "drilldown_ms": timing.get("drilldown_total_ms", 0.0),
                        "total_ms": timing.get("search_total_ms", timing.get("total_ms", 0.0)),
                        "result_count": len(result["results"]),
                        "top_document": result["results"][0]["filename"] if result["results"] else None,
                    }
                    print(json.dumps(row, ensure_ascii=False))
        finally:
            pool.close()


if __name__ == "__main__":
    main()
