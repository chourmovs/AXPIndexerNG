#!/usr/bin/env python3
"""Opt-in exact-vector audit on an existing read-only catalog (JSON output).

Paired variants share a WAL snapshot, with alternating order. A new connection
is not a cold disk cache. No OS cache flush, daemon control or corpus mutation.
"""
import argparse
import json
import platform
import sqlite3
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for folder in ("shared", "daemon", "client"):
    sys.path.insert(0, str(ROOT / folder))

from axp_core.database import configure_search_reader, connect  # noqa: E402
from axp_core.vectors import (developer_diagnostics, enrich, knn, read_snapshot, search)  # noqa: E402


def observation():
    """Process scope: CPU and physical read bytes, not SQLite page-cache misses."""
    import psutil

    process = psutil.Process()
    cpu = process.cpu_times()
    try:
        io = process.io_counters()
        read_bytes = io.read_bytes
    except (AttributeError, psutil.Error):
        read_bytes = None
    return {"cpu_ms": (cpu.user + cpu.system) * 1000, "read_bytes": read_bytes}


def daemon_observation():
    import psutil

    processes = []
    for process in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            arguments = process.info["cmdline"] or []
            if not any("axp_daemon" in arg or arg.endswith("AXPIndexerDaemon.pyw") for arg in arguments):
                continue
            cpu = process.cpu_times()
            try:
                read_bytes = process.io_counters().read_bytes
            except (AttributeError, psutil.Error):
                read_bytes = None
            processes.append({"pid": process.pid, "name": process.info["name"],
                              "cpu_ms": (cpu.user + cpu.system) * 1000, "read_bytes": read_bytes})
        except psutil.Error:
            continue
    return processes


def measured(call):
    before = observation()
    tick = time.perf_counter()
    result = call()
    elapsed = (time.perf_counter() - tick) * 1000
    after = observation()
    return result, {"wall_ms": elapsed, **{key: after[key] - before[key]
        if before[key] is not None and after[key] is not None else None for key in before}}


def summary(values):
    values = sorted(values)
    quartiles = statistics.quantiles(values, n=4, method="inclusive") if len(values) > 1 else [values[0]] * 3
    return {"n": len(values), "median": statistics.median(values), "min": values[0], "max": values[-1],
            "iqr": quartiles[2] - quartiles[0]}


class SyntheticEmbedder:
    model_id = "synthetic"
    dimension = 384
    distance_metric = "cosine"

    def embed_query(self, query):
        import math
        v = [float("pressure" in query.lower()), float("storage" in query.lower()), float("DOMINO" in query)]
        v += [float(not any(v))] + [0.] * 380
        norm = math.sqrt(sum(x * x for x in v))
        return [x / norm for x in v]


def run(args):
    daemon_start = daemon_observation()
    con = connect(args.db, readonly=True)
    try:
        configure_search_reader(con, include_diagnostics=False)
        embedder = None
        if args.vectors_json:
            queries = json.loads(args.vectors_json.read_text(encoding="utf-8"))
        else:
            if args.synthetic:
                embedder = SyntheticEmbedder()
                from axp_core.metadata import validate_index_signature
                validate_index_signature(con, embedder.model_id, embedder.dimension, embedder.distance_metric)
            else:
                from axp_daemon.embeddings import embedder_for_index
                embedder = embedder_for_index(con)
            queries = [{"query": query, "vector": embedder.embed_query(query)} for query in
                       (args.query or ["DOMINO", "reactor pressure", "DOMINO pressure",
                                       "DOMINO storage pressure", "unobtainium calibration"])]
        if args.full_search and embedder is None:
            raise ValueError("--full-search requires the catalog embedder or --synthetic")
        if not queries:
            raise ValueError("At least one query/vector pair is required")
        rows = []
        # Do not run EQP/counts before measuring the first vector invocation.
        for entry in queries:
            query, vector = entry["query"], entry["vector"]
            for rep in range(args.repeats + 1):
                with read_snapshot(con):
                    timings = {}
                    outputs = {}
                    # The first variant necessarily warms the second. Rotate it.
                    order = ["combined", "split"] if rep % 2 == 0 else ["split", "combined"]
                    for position, variant in enumerate(order):
                        local = {}
                        if variant == "combined":
                            result, metrics = measured(lambda: search(con, vector, args.top_k, timings=local))
                        else:
                            candidates, knn_metrics = measured(lambda: knn(con, vector, args.top_k, timings=local))
                            result, enrichment_metrics = measured(lambda: enrich(con, candidates, timings=local))
                            metrics = {"wall_ms": knn_metrics["wall_ms"] + enrichment_metrics["wall_ms"],
                                       "knn": knn_metrics, "enrichment": enrichment_metrics}
                        outputs[variant] = result
                        timings[variant] = {**metrics, **local, "position": position}
                    # Distances, all metadata and native tie ordering must agree.
                    equal = outputs["combined"] == outputs["split"]
                    row = {"query": query, "rep": rep, "phase": "first_pair" if rep == 0 else "repeat",
                           "variant_order": order, "equivalent": equal, "timings": timings,
                           "ordered_chunk_ids": [x["chunk_id"] for x in outputs["combined"]],
                           "distances": [x["vector_distance"] for x in outputs["combined"]]}
                    if args.full_search:
                        from axp_client.search import search as full_search
                        result, metrics = measured(lambda: full_search(con, embedder, query, explain=True))
                        row["server_search"] = {**metrics, "timings": result["timings"],
                            "ordered_documents": [x["document_id"] for x in result["results"]],
                            "lexical_candidate_count": result["candidate_counts"]["lexical"]}
                        from axp_core.fts import search as lexical_search
                        lexical, lexical_metrics = measured(lambda: lexical_search(con, query, 100))
                        preview = list(dict.fromkeys(x["document_id"] for x in lexical))[:3]
                        final_ids = row["server_search"]["ordered_documents"]
                        row["lexical_preview_audit"] = {**lexical_metrics, "ordered_documents": preview,
                            "overlap_with_final": [value for value in preview if value in final_ids],
                            "note": "after full search on warm connection; overlap is not a relevance judgment"}
                    rows.append(row)
                    if not equal:
                        raise AssertionError("Combined/split mismatch; do not deploy a rewrite")
        diagnostics = developer_diagnostics(con, queries[0]["vector"], args.top_k)
        aggregates = {}
        for entry in queries:
            sample = [r for r in rows if r["query"] == entry["query"] and r["phase"] == "repeat"]
            aggregates[entry["query"]] = {variant: summary([r["timings"][variant]["wall_ms"] for r in sample])
                                           for variant in ("combined", "split")}
            for stage in ("knn", "enrichment"):
                aggregates[entry["query"]][stage] = summary([r["timings"]["split"][stage]["wall_ms"] for r in sample])
            if args.full_search:
                aggregates[entry["query"]]["server_search"] = summary([r["server_search"]["wall_ms"] for r in sample])
        return {"catalog": str(args.db.resolve()), "synthetic": args.synthetic,
                "environment_label": args.environment_label, "platform": platform.platform(),
                "daemon_processes_start": daemon_start, "daemon_processes_end": daemon_observation(),
                "sqlite": sqlite3.sqlite_version, "diagnostics": diagnostics,
                "sqlite_settings": {name: con.execute(f"PRAGMA {name}").fetchone()[0]
                    for name in ("cache_size", "mmap_size", "temp_store", "page_size", "journal_mode", "busy_timeout")},
                "measurement_notes": ["one read-only connection; first_pair is not an OS cold-cache claim",
                    "paired variants share a transaction; order alternates; no ANN",
                    "execute includes disk reads, SQLite/extension work, joins and waits",
                    "process CPU/read_bytes are not per-query SQLite cache statistics",
                    "returned rows do not measure vectors visited; no browser measurement here",
                    "server_search runs after paired variants; uses a warmed connection"],
                "rows": rows, "summary": aggregates}
    finally:
        con.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=7, help="warm repetitions, in addition to first_pair")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--query", action="append")
    parser.add_argument("--vectors-json", type=Path, help='JSON list of {"query":str,"vector":[float,...]}')
    parser.add_argument("--synthetic", action="store_true", help="only for benchmark_recovery.py fixture")
    parser.add_argument("--full-search", action="store_true")
    parser.add_argument("--environment-label", default="daemon state unobserved")
    args = parser.parse_args()
    if args.repeats < 1 or not 1 <= args.top_k <= 500:
        parser.error("repeats >= 1 and 1 <= top-k <= 500 required")
    if args.synthetic and args.vectors_json:
        parser.error("choose --synthetic or --vectors-json")
    if args.output.resolve() == args.db.resolve():
        parser.error("output must differ from the catalog")
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
