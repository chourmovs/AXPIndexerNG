#!/usr/bin/env python3
"""Read-only ANN qualification against an existing AXP SQLite catalog.

Builds one temporary USearch HNSW index from the stored float32 vectors.
Compares exactly the same cohort against exhaustive cosine ground truth.
When the entire corpus is used, also measures the native sqlite-vec KNN.
No live database writes, no embeddings regenerated, no production switch.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import random
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "shared"))

from axp_core.ann import UsearchHNSW  # noqa: E402
from axp_core.ann_metrics import exact_cosine_topk, recall_at_k, select_by_position  # noqa: E402
from axp_core.database import connect  # noqa: E402
from axp_core.metadata import read_index_signature  # noqa: E402
from axp_core.performance import nearest_rank  # noqa: E402
from axp_core.vectors import knn  # noqa: E402


def distribution(values):
    return {"n": len(values), **({
        "p50": nearest_rank(values, 50), "p95": nearest_rank(values, 95),
        "p99": nearest_rank(values, 99), "min": min(values), "max": max(values),
    } if values else {})}


def read_queries(path, dimension):
    queries = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(queries, list) or not queries:
        raise ValueError("queries JSON must be a nonempty list")
    result = []
    for i, row in enumerate(queries):
        if not isinstance(row, dict) or len(row.get("vector", [])) != dimension:
            raise ValueError(f"query {i} has wrong dimension")
        result.append((str(row.get("query") or f"query-{i}"), row["vector"]))
    return result


def memory_rss():
    try:
        import psutil
        return int(psutil.Process().memory_info().rss)
    except ImportError:
        return None


def run(args):
    import numpy as np

    if not args.db.is_file():
        raise ValueError("catalog does not exist")
    if args.top_k < 1 or args.max_vectors < 1 or args.batch_size < 1 or args.query_count < 1:
        raise ValueError("top-k, max-vectors, batch-size and query-count must be positive")
    if args.index_output and args.index_output.exists():
        raise FileExistsError("refusing to overwrite existing ANN index")
    with connect(args.db, readonly=True) as con:
        signature = read_index_signature(con)
        dimension = int(signature["embedding_dimension"])
        if signature["distance_metric"] != "cosine":
            raise ValueError("only cosine indexes are currently supported")
        total = int(con.execute("SELECT count(*) FROM chunk_vectors").fetchone()[0])
        if total < 1:
            raise ValueError("catalog has no vectors")
        selected = min(args.max_vectors, total)
        rng = random.Random(args.seed)
        query_samples = []
        started = time.perf_counter()
        # Windows keeps mapped files locked until their mappings are explicitly closed.
        # TemporaryDirectory would attempt deletion before local memmap references expire.
        scratch = Path(tempfile.mkdtemp(prefix="axp-ann-benchmark-"))
        keys = matrix = index = reopened = None
        try:
            keys = np.memmap(scratch / "ids.dat", mode="w+", dtype=np.int64, shape=(selected,))
            matrix = np.memmap(Path(scratch) / "vectors.dat", mode="w+", dtype=np.float32,
                               shape=(selected, dimension))
            position = 0
            stored = 0
            for row in con.execute("SELECT rowid, embedding FROM chunk_vectors ORDER BY rowid"):
                position += 1
                if not select_by_position(position, total, selected):
                    continue
                vec = np.frombuffer(row["embedding"], dtype="<f4")
                if len(vec) != dimension or not np.isfinite(vec).all():
                    raise ValueError(f"invalid vector at rowid {row['rowid']}")
                norm = float(np.linalg.norm(vec))
                if norm <= 1e-12:
                    raise ValueError(f"zero embedding at rowid {row['rowid']}")
                keys[stored] = int(row["rowid"])
                matrix[stored] = vec / norm
                if args.queries_json is None:
                    sample = (f"indexed-chunk-{int(row['rowid'])}", np.asarray(vec).copy())
                    if stored < args.query_count:
                        query_samples.append(sample)
                    else:
                        j = rng.randrange(stored + 1)
                        if j < args.query_count:
                            query_samples[j] = sample
                stored += 1
            if stored != selected:
                raise RuntimeError(f"sampling mismatch: expected {selected}, got {stored}")
            extract_ms = (time.perf_counter() - started) * 1000
            query_samples = (read_queries(args.queries_json, dimension)
                             if args.queries_json is not None else query_samples)
            if not query_samples:
                raise ValueError("no available query vectors")
            index = UsearchHNSW(dimension, dtype=args.dtype, connectivity=args.connectivity,
                expansion_add=args.expansion_add, expansion_search=args.expansion_search)
            started = time.perf_counter()
            for start in range(0, selected, args.batch_size):
                end = min(selected, start + args.batch_size)
                index.add(np.asarray(keys[start:end], dtype=np.uint64),
                          np.asarray(matrix[start:end], dtype=np.float32))
            build_ms = (time.perf_counter() - started) * 1000
            if index.size != selected:
                raise RuntimeError("HNSW count mismatch; refusing result")
            index_memory = index.memory_usage
            index_file = Path(scratch) / "candidate.usearch"
            started = time.perf_counter()
            index.save(index_file)
            save_ms = (time.perf_counter() - started) * 1000
            size_bytes = index_file.stat().st_size
            # Saved index must be usable after reopen, before any recommendation.
            try:
                from usearch.index import Index
                reopened = Index(ndim=dimension, metric="cos", dtype=args.dtype,
                                 path=str(index_file), view=True)
                assert len(reopened.search(np.asarray(query_samples[0][1], dtype=np.float32),
                                           min(args.top_k, selected)).keys) > 0
                persistence_ok = True
                del reopened
            except Exception as exc:
                raise RuntimeError("USearch memory-mapped reopen test failed") from exc
            if args.index_output is not None:
                args.index_output.parent.mkdir(parents=True, exist_ok=True)
                index.save(args.index_output)
            cases = []
            for label, vector in query_samples:
                query = np.asarray(vector, dtype=np.float32)
                started = time.perf_counter()
                truth = exact_cosine_topk(matrix, keys, query, args.top_k)
                exact_ms = (time.perf_counter() - started) * 1000
                started = time.perf_counter()
                found = index.search(query, min(args.top_k, selected))
                ann_ms = (time.perf_counter() - started) * 1000
                actual = [key for key, _ in found]
                if len(actual) != min(selected, args.top_k) or len(set(actual)) != len(actual):
                    raise RuntimeError("ANN returned missing or duplicate IDs")
                row = {"query": label, "exact_ms": exact_ms, "ann_ms": ann_ms,
                       "recall_at_k": recall_at_k(truth, actual, args.top_k)}
                if selected == total:
                    started = time.perf_counter()
                    native_ids = [int(hit["chunk_id"]) for hit in knn(con, query, min(args.top_k, 500))]
                    row["sqlite_vec_exact_ms"] = (time.perf_counter() - started) * 1000
                    row["sqlite_vec_recall_vs_numpy"] = recall_at_k(truth, native_ids, args.top_k)
                cases.append(row)
            return {
                "benchmark": "USearch HNSW versus exhaustive cosine (no live index switch)",
                "source": str(args.db.resolve()),
                "catalog_total_vectors": total, "sample_vectors": selected,
                "sample_strategy": "deterministic position-stratified",
                "embedding_model_id": signature["embedding_model_id"],
                "dimension": dimension, "metric": "cos", "dtype": args.dtype,
                "seed": args.seed, "top_k": args.top_k, "query_source": (
                    "external query embeddings" if args.queries_json else "sampled indexed vectors"),
                "queries": len(query_samples), "connectivity": args.connectivity,
                "expansion_add": args.expansion_add, "expansion_search": args.expansion_search,
                "build_ms": build_ms, "extract_ms": extract_ms, "save_ms": save_ms,
                "index_bytes": size_bytes, "index_memory_bytes": index_memory,
                "rss_bytes": memory_rss(), "persisted_reopen_ok": persistence_ok,
                "os": platform.platform(), "blas_threads": os.getenv("OPENBLAS_NUM_THREADS", "unknown"),
                "latency": {
                    "ann_ms": distribution([case["ann_ms"] for case in cases]),
                    "numpy_exact_ms": distribution([case["exact_ms"] for case in cases]),
                    "sqlite_vec_exact_ms": distribution([case["sqlite_vec_exact_ms"] for case in cases
                                                         if "sqlite_vec_exact_ms" in case]),
                },
                "recall_at_k_mean": sum(case["recall_at_k"] for case in cases) / len(cases),
                "recall_at_k_min": min(case["recall_at_k"] for case in cases),
                "cases": cases,
                "limitations": [
                    "Sampled indexed-vector queries contain a self-match: use --queries-json for user-query recall",
                    "Approximate top-k ties may differ even at equal distance",
                    "Numpy and USearch search the same sampled population; native SQLite comparison only for full corpus",
                    "Native sqlite-vec k is bounded to 500; keep --top-k <= 500",
                    "Memory, CPU contention, construction and IO affect measured wall latency",
                    "No SQLite Vec1 implementation or winner decision is claimed in this PR",
                ],
            }

        finally:
            # Release the native memory-mapped USearch view before deleting the workspace.
            # It is essential to close NumPy mappings, not merely to flush their contents.
            del reopened, index
            gc.collect()
            for mapped in (matrix, keys):
                if mapped is None:
                    continue
                mapped.flush()
                mapping = getattr(mapped, "_mmap", None)
                if mapping is not None and not mapping.closed:
                    mapping.close()
            matrix = keys = None
            gc.collect()
            shutil.rmtree(scratch)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--max-vectors", type=int, default=50000)
    parser.add_argument("--queries-json", type=Path, help='optional JSON list of query embeddings')
    parser.add_argument("--query-count", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--connectivity", type=int, default=24)
    parser.add_argument("--expansion-add", type=int, default=128)
    parser.add_argument("--expansion-search", type=int, default=64)
    parser.add_argument("--dtype", choices=("f32", "f16", "bf16"), default="f32")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--index-output", type=Path,
                        help="optional persistent sidecar path; defaults to ephemeral temporary directory")
    parser.add_argument("--output", type=Path, help="optional JSON output file; never overwrites")
    args = parser.parse_args()
    if args.top_k > 500 or args.connectivity < 2 or args.expansion_add < 1 or args.expansion_search < 1:
        parser.error("top-k <= 500; connectivity >= 2; ef-add and ef-search > 0")
    if args.output and args.output.exists():
        parser.error("refusing to overwrite output")
    result = run(args)
    text = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
