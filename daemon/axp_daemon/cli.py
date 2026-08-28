import argparse
import json
import os

from axp_core.database import capability_report, connect
from axp_core.metadata import ensure_index_signature
from axp_core.runtime import read_json, runtime_paths
from axp_core.sources import add_source, disable_source, enable_source, get_source, list_sources, remove_source

from .embeddings import Embedder
from .indexer import scan, scan_source
from .service import _safe_reindex, run_daemon, send_control


def _embedding_args(parser):
    parser.add_argument("--model-cache")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--embedding-profile", choices=("balanced", "quality"), default="balanced")
    parser.add_argument("--embedding-batch-size", type=int, default=64)


def _embedder(args):
    return Embedder(args.embedding_profile, cache_dir=args.model_cache or os.getenv("FASTEMBED_CACHE_PATH"),
                    local_only=not args.allow_download)


def _source_json(row):
    value = dict(row)
    value["enabled"] = bool(value["enabled"])
    value["recursive"] = bool(value["recursive"])
    return value


def main(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("health", "status"):
        command = sub.add_parser(name)
        command.add_argument("--db", required=True)
    for name in ("scan", "reindex"):
        command = sub.add_parser(name)
        command.add_argument("--db", required=True)
        command.add_argument("--root", required=True)
        _embedding_args(command)
    run = sub.add_parser("run")
    run.add_argument("--db", required=True)
    run.add_argument("--model-cache")
    run.add_argument("--allow-download", action="store_true")
    run.add_argument("--embedding-profile", choices=("balanced", "quality"), default="balanced")
    run.add_argument("--embedding-batch-size", type=int, default=64)
    run.add_argument("--scan-interval", type=int, default=300)
    run.add_argument("--model-download-retry", type=int, default=60)
    source = sub.add_parser("source")
    source_sub = source.add_subparsers(dest="source_cmd", required=True)
    for name in ("list", "add", "enable", "disable", "remove"):
        command = source_sub.add_parser(name)
        command.add_argument("--db", required=True)
        if name == "add":
            command.add_argument("--path", required=True)
            command.add_argument("--label")
            command.add_argument("--allow-subsuming", action="store_true")
        elif name != "list":
            command.add_argument("--id", type=int, required=True)
    scan_one = sub.add_parser("scan-source")
    scan_one.add_argument("--db", required=True)
    scan_one.add_argument("--id", type=int, required=True)
    _embedding_args(scan_one)
    control = sub.add_parser("control")
    control.add_argument("action", choices=("scan", "pause", "resume", "stop"))
    args = parser.parse_args(argv)

    if args.cmd == "control":
        print(json.dumps(send_control(args.action)))
        return
    if args.cmd == "run":
        value = run_daemon(
            args.db, args.model_cache or os.getenv("FASTEMBED_CACHE_PATH"), args.embedding_profile,
            max(1, args.scan_interval), max(1, args.embedding_batch_size), args.allow_download,
            max(1, args.model_download_retry),
        )
        print(json.dumps(value))
        return
    con = connect(args.db)
    if args.cmd == "health":
        print(json.dumps(capability_report(con)))
        return
    if args.cmd == "status":
        print(json.dumps({
            "documents": con.execute("SELECT count(*) FROM documents").fetchone()[0],
            "chunks": con.execute("SELECT count(*) FROM chunks").fetchone()[0],
            "sources": [_source_json(row) for row in list_sources(con)],
            "runtime": read_json(runtime_paths()["state"], {}),
        }, ensure_ascii=False))
        return
    if args.cmd == "source":
        if args.source_cmd == "list":
            value = [_source_json(row) for row in list_sources(con)]
        elif args.source_cmd == "add":
            value = _source_json(add_source(con, args.path, args.label, allow_subsume=args.allow_subsuming))
        elif args.source_cmd == "enable":
            value = _source_json(enable_source(con, args.id))
        elif args.source_cmd == "disable":
            value = _source_json(disable_source(con, args.id))
        else:
            value = _source_json(remove_source(con, args.id))
        print(json.dumps(value, ensure_ascii=False))
        return

    embedder = _embedder(args)
    con.close()
    con = connect(args.db, dimension=embedder.dimension)
    ensure_index_signature(con, embedder.model_id, embedder.dimension, embedder.distance_metric)
    if args.cmd == "scan-source":
        get_source(con, args.id)
        value = scan_source(con, args.id, embedder, embedding_batch_size=args.embedding_batch_size)
    elif args.cmd == "reindex":
        from axp_core.sources import normalize_source_path

        display, key = normalize_source_path(args.root)
        row = con.execute("SELECT id FROM sources WHERE path_key=?", (key,)).fetchone()
        source_id = row[0] if row else add_source(con, display)["id"]
        _safe_reindex(con, source_id)
        value = scan_source(con, source_id, embedder, embedding_batch_size=args.embedding_batch_size)
    else:
        value = scan(con, args.root, embedder, embedding_batch_size=args.embedding_batch_size)
    print(json.dumps(value, ensure_ascii=False))
