import argparse
import json
import os

from axp_core.database import capability_report, connect
from axp_core.hybrid import SearchConfig
from axp_daemon.embeddings import embedder_for_index

from .reranker import Reranker
from .search import search
from .server import serve


def main(argv=None):
    p = argparse.ArgumentParser()
    s = p.add_subparsers(dest="cmd", required=True)
    for cmd in ("health", "search", "serve"):
        q = s.add_parser(cmd)
        q.add_argument("--db", required=True)
        if cmd == "search":
            q.add_argument("--query", required=True)
            q.add_argument(
                "--profile",
                choices=("fast", "hybrid", "quality"),
                default="hybrid",
                help="retrieval profile; quality adds ColBERT reranking (independent of the index embedding profile)",
            )
            q.add_argument("--limit", type=int, default=20)
            q.add_argument("--explain", action="store_true")
            q.add_argument("--lexical-candidates", type=int, default=100)
            q.add_argument("--vector-candidates", type=int, default=100)
            q.add_argument("--rerank-candidates", type=int, default=30)
        if cmd == "serve":
            q.add_argument("--host", default="127.0.0.1")
            q.add_argument("--port", type=int, default=8765)
    a = p.parse_args(argv)
    if a.cmd == "health":
        print(json.dumps(capability_report(connect(a.db))))
        return
    con = connect(a.db, readonly=True)
    e = embedder_for_index(con, cache_dir=os.getenv("FASTEMBED_CACHE_PATH"), local_only=True)
    if a.cmd == "search":
        config = SearchConfig(a.lexical_candidates, a.vector_candidates, a.rerank_candidates)
        reranker = Reranker(cache_dir=os.getenv("FASTEMBED_CACHE_PATH")) if a.profile == "quality" else None
        value = search(con, e, a.query, a.limit, profile=a.profile, explain=a.explain, reranker=reranker, config=config)
        print(json.dumps(value, ensure_ascii=False))
        return
    serve(a.db, e, a.host, a.port)
