#!/usr/bin/env python3
"""Compare the legacy and progressive retrieval paths against an existing index."""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT, ROOT / "shared", ROOT / "daemon", ROOT / "client"):
    sys.path.insert(0, str(folder))

from axp_client.rag.spiral import SpiralRetriever  # noqa: E402
from axp_client.rag.retrieval import retrieve_rag_candidates  # noqa: E402
from axp_client.search import search  # noqa: E402
from axp_core.database import connect  # noqa: E402
from axp_daemon.embeddings import embedder_for_index  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()
    with connect(args.db, readonly=True) as con:
        embedder = embedder_for_index(con)
        for run in range(max(1, args.repeat)):
            started = time.perf_counter()
            legacy = retrieve_rag_candidates(con, embedder, args.query, search_fn=search)
            legacy_ms = (time.perf_counter() - started) * 1000
            spiral = SpiralRetriever(search_fn=search).retrieve(con, embedder, args.query)
            top_legacy = legacy.ranked_documents[0].get("filename") if legacy.ranked_documents else "-"
            top_spiral = (spiral.retrieval.ranked_documents[0].get("filename")
                          if spiral.retrieval.ranked_documents else "-")
            print(f"Query: {args.query} (run {run + 1})")
            print(f"Legacy\n  retrieval       {legacy_ms:.1f} ms\n  top document    {top_legacy}")
            print(f"  answerability   {spiral.decision.answerable}")
            print(f"  candidates      {len(legacy.candidates)}")
            print(f"Spiral\n  retrieval       {spiral.retrieval.timings['retrieval_ms']:.1f} ms")
            print(f"  stop             {spiral.stage}")
            print(f"  global fallback  {'yes' if spiral.global_fallback_used else 'no'}")
            print(f"  query embeddings 1\n  reader reused    n/a\n  top document     {top_spiral}")
            print(f"  answerability   {spiral.decision.answerable}")
            print(f"  candidates      {len(spiral.retrieval.candidates)}")
            print(f"  timings          {spiral.retrieval.timings}")


if __name__ == "__main__":
    main()
