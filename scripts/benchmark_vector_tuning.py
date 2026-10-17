#!/usr/bin/env python3
"""Read-only, seeded SQLite/top-k sweep. Writes raw JSON, stage CSV and Markdown."""
import argparse
import csv
import itertools
import json
import random
from pathlib import Path
from types import SimpleNamespace

try:
    from scripts.benchmark_vectors import run
except ModuleNotFoundError:
    from benchmark_vectors import run


def configurations(top_k, cache_mib, mmap_mib):
    return [{"top_k": k, "cache_mib": cache, "mmap_mib": mmap}
            for k, cache, mmap in itertools.product(top_k, cache_mib, mmap_mib)]


def export(directory, records):
    fields = ["round", "config", "query", "stage", "n", "p50_ms", "p95_ms", "min_ms", "max_ms",
              "same_k_order_and_distances_unchanged"]
    table = []
    references = {}
    for record in records:
        result, config = record["result"], record["config"]
        for query, stages in result["summary"].items():
            samples = [r for r in result["rows"] if r["query"] == query]
            key = (config["top_k"], query)
            signatures = [(r["ordered_chunk_ids"], r["distances"]) for r in samples]
            reference = references.setdefault(key, signatures[0])
            unchanged = all(signature == reference for signature in signatures)
            for stage, stats in stages.items():
                table.append(dict(zip(fields, [record["round"], json.dumps(config, sort_keys=True), query,
                    stage, stats["n"], stats["p50"], stats["p95"], stats["min"], stats["max"], unchanged])))
    with (directory / "stages.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(table)
    lines = ["# Mesures vectorielles", "", "p95 : rang le plus proche ; premiers couples exclus des statistiques.",
             "Comparer à top-k identique. La stabilité des résultats ne mesure pas leur pertinence.", "",
             "| Tour | Configuration | Requête | Étape | p50 ms | p95 ms | Résultats stables |",
             "|---|---|---|---|---:|---:|---|"]
    for row in table:
        cells = [row[key] for key in fields[:4]] + [f'{row["p50_ms"]:.2f}', f'{row["p95_ms"]:.2f}',
                                                     row[fields[-1]]]
        lines.append("| " + " | ".join(str(v).replace("|", "\\|").replace("\n", " ") for v in cells) + " |")
    (directory / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--vectors-json", type=Path, required=True,
                        help='Fixed list of {"query": str, "vector": [float,...]} from the catalog model')
    parser.add_argument("--top-k", type=int, nargs="+", default=[100])
    parser.add_argument("--cache-mib", type=int, nargs="+", default=[64])
    parser.add_argument("--mmap-mib", type=int, nargs="+", default=[256])
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--environment-label", default="daemon state unobserved")
    args = parser.parse_args()
    if (args.repeats < 1 or args.rounds < 1 or any(not 1 <= k <= 500 for k in args.top_k)
            or any(v <= 0 for v in args.cache_mib) or any(v < 0 for v in args.mmap_mib)):
        parser.error("repeats/rounds/cache > 0, mmap >= 0, 1 <= top-k <= 500 required")
    if not args.db.is_file():
        parser.error("catalog does not exist")
    entries = json.loads(args.vectors_json.read_text(encoding="utf-8"))
    if not entries or len({e["query"] for e in entries}) != len(entries):
        parser.error("provide a nonempty list with unique query labels")
    if args.output_dir.resolve() == args.db.resolve() or args.db.resolve().is_relative_to(args.output_dir.resolve()):
        parser.error("use a separate output directory outside the catalog's parent tree")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    rng = random.Random(args.seed)
    records = []
    matrix = configurations(args.top_k, args.cache_mib, args.mmap_mib)
    for round_no in range(args.rounds):
        order = list(matrix)
        rng.shuffle(order)
        for config in order:
            print(f"round={round_no + 1} config={config}", flush=True)
            settings = {"cache_size": -config["cache_mib"] * 1024,
                        "mmap_size": config["mmap_mib"] * 1024**2}
            options = SimpleNamespace(db=args.db, vectors_json=args.vectors_json, synthetic=False,
                full_search=False, query=None, repeats=args.repeats, top_k=config["top_k"],
                environment_label=args.environment_label, sqlite_settings=settings)
            result = run(options)
            record = {"round": round_no + 1, "config": config, "result": result}
            records.append(record)
            # Persist every completed configuration so a long sweep can be interrupted.
            (args.output_dir / "raw.json").write_text(json.dumps({"seed": args.seed, "records": records},
                ensure_ascii=False, indent=2), encoding="utf-8")
            export(args.output_dir, records)
    print(f"Report: {args.output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
