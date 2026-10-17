#!/usr/bin/env python3
"""Measure HTTP search latency under bounded client concurrency (standard library)."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import urlopen

try:
    from scripts.benchmark_vectors import summary
except ModuleNotFoundError:
    from benchmark_vectors import summary


def request(base, query, timeout):
    start = time.perf_counter()
    try:
        with urlopen(base.rstrip("/") + "/api/search?" + urlencode({"q": query, "explain": "1"}),
                     timeout=timeout) as response:
            payload = json.load(response)
        return {"query": query, "ok": True, "elapsed_ms": (time.perf_counter() - start) * 1000,
                "timings": payload.get("timings"), "candidate_counts": payload.get("candidate_counts")}
    except (HTTPError, URLError, TimeoutError, OSError, ValueError) as exc:
        return {"query": query, "ok": False, "elapsed_ms": (time.perf_counter() - start) * 1000,
                "error": str(exc)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    parser.add_argument("--queries-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--warmups", type=int, default=1, help="unmeasured sequential passes over queries")
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()
    if min(args.concurrency) < 1 or max(args.concurrency) > 64 or args.repeats < 1 or args.warmups < 0 or args.timeout <= 0:
        parser.error("concurrency 1..64, repeats/timeout > 0, warmups >= 0 required")
    if urlsplit(args.url).scheme not in {"http", "https"}:
        parser.error("HTTP(S) URL required")
    queries = json.loads(args.queries_json.read_text(encoding="utf-8"))
    if not isinstance(queries, list) or not queries or any(not isinstance(q, str) or not q.strip() for q in queries):
        parser.error("queries-json must contain a nonempty list of query strings")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Reserve the output before issuing requests; do not overwrite a catalog or prior report.
    with args.output.open("x", encoding="utf-8") as stream:
        warmups = [request(args.url, q, args.timeout) for _ in range(args.warmups) for q in queries]
        results = []
        for concurrency in args.concurrency:
            start = time.perf_counter()
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                rows = list(pool.map(lambda q: request(args.url, q, args.timeout), queries * args.repeats))
            seconds = time.perf_counter() - start
            successful = [r["elapsed_ms"] for r in rows if r["ok"]]
            result = {"concurrency": concurrency, "requests": len(rows), "errors": len(rows) - len(successful),
                      "successful_requests_per_second": len(successful) / seconds,
                      "latency_ms": summary(successful) if successful else None, "rows": rows}
            results.append(result)
            print(json.dumps({k: v for k, v in result.items() if k != "rows"}), flush=True)
        json.dump({"url": args.url, "warmups": warmups, "results": results,
                   "notes": ["closed-loop client concurrency; client elapsed includes HTTP and JSON parsing",
                             "server caches remain warm across levels; not browser paint latency",
                             "p95 uses nearest rank; errors are reported separately"]}, stream, indent=2)


if __name__ == "__main__":
    main()
