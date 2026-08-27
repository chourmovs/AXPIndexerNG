import argparse
import json
import os

from axp_core.database import capability_report, connect
from axp_daemon.embeddings import Embedder

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
        if cmd == "serve":
            q.add_argument("--host", default="127.0.0.1")
            q.add_argument("--port", type=int, default=8765)
    a = p.parse_args(argv)
    if a.cmd == "health":
        print(json.dumps(capability_report(connect(a.db))))
        return
    e = Embedder(cache_dir=os.getenv("FASTEMBED_CACHE_PATH"), local_only=True)
    if a.cmd == "search":
        print(json.dumps(search(connect(a.db, readonly=True), e, a.query), ensure_ascii=False))
        return
    serve(a.db, e, a.host, a.port)
