"""Field qualification for CPU-query/OpenVINO-document embedding compatibility.

The report contains metrics only: sampled document text is never printed or written.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sqlite3
import time
from pathlib import Path

from axp_core.runtime import atomic_write_json, data_dir
from axp_daemon.embedding_backends.cpu import CPUEmbeddingBackend
from axp_daemon.embedding_backends.intel_openvino import IntelOpenVINOBackend


DEFAULT_TEXTS = [
    "AXP indexing qualification short sentence.",
    "Qualification multilingue en français pour la recherche documentaire.",
    "Pump P-104 pressure=42.75 bar; drawing AX-2026-009.pdf.",
    "Technical procedure: isolate valve XV-17, verify 24 VDC, then record lot AB12-9087.",
] * 64


def _sample(path, count):
    if not path:
        return DEFAULT_TEXTS[:count]
    con = sqlite3.connect(path)
    try:
        # Random IDs avoid retrieving/logging more corporate content than the requested bound.
        rows = con.execute("SELECT text FROM chunks WHERE length(trim(text)) > 0 ORDER BY random() LIMIT ?",
                           (count,)).fetchall()
    finally:
        con.close()
    if not rows:
        raise RuntimeError("sample database contains no chunk text")
    values = [row[0] for row in rows]
    random.Random(642).shuffle(values)
    return values


def _cosine(left, right):
    denominator = math.sqrt(sum(x * x for x in left)) * math.sqrt(sum(x * x for x in right))
    return sum(x * y for x, y in zip(left, right, strict=True)) / denominator if denominator else 0.0


def _rankings(queries, documents, limit=10):
    return [sorted(range(len(documents)), key=lambda index: _cosine(query, documents[index]), reverse=True)[:limit]
            for query in queries]


def qualify(*, sample_db=None, count=256, batch_size=64, model="balanced", runtime_dir=None):
    texts = _sample(sample_db, max(16, min(count, 512)))
    root = Path(runtime_dir or data_dir() / "runtime" / "intel-embedding")
    cache = data_dir() / "cache" / "openvino"
    started = time.perf_counter()
    cpu = CPUEmbeddingBackend(model, batch_size=batch_size)
    cpu_init_ms = (time.perf_counter() - started) * 1000
    started = time.perf_counter()
    gpu = IntelOpenVINOBackend(model, runtime_dir=root, cache_dir=cache, batch_size=batch_size,
                               require_qualification=False)
    gpu_init_ms = (time.perf_counter() - started) * 1000
    started = time.perf_counter()
    cpu_vectors = cpu.embed_documents(texts)
    cpu_ms = (time.perf_counter() - started) * 1000
    started = time.perf_counter()
    gpu_vectors = gpu.embed_documents(texts)
    gpu_ms = (time.perf_counter() - started) * 1000
    if len(cpu_vectors) != len(gpu_vectors):
        raise RuntimeError("backend returned a different vector count")
    cosines, max_deltas, mean_deltas = [], [], []
    dimensions_match = True
    for reference, candidate in zip(cpu_vectors, gpu_vectors, strict=True):
        dimensions_match &= len(reference) == len(candidate) == cpu.dimension
        deltas = [abs(x - y) for x, y in zip(reference, candidate)]
        cosines.append(_cosine(reference, candidate))
        max_deltas.append(max(deltas, default=0.0))
        mean_deltas.append(sum(deltas) / len(deltas) if deltas else 0.0)
    query_count = min(100, len(texts))
    cpu_queries = [cpu.embed_query(text) for text in texts[:query_count]]
    reference_ranks = _rankings(cpu_queries, cpu_vectors)
    cross_ranks = _rankings(cpu_queries, gpu_vectors)
    top1 = sum(a[0] == b[0] for a, b in zip(reference_ranks, cross_ranks, strict=True)) / query_count
    recall10 = sum(len(set(a) & set(b)) / len(a) for a, b in zip(reference_ranks, cross_ranks, strict=True)) / query_count
    recall5 = sum(len(set(a[:5]) & set(b[:5])) / min(5, len(a))
                  for a, b in zip(reference_ranks, cross_ranks, strict=True)) / query_count
    cpu_rate = len(texts) / (cpu_ms / 1000) if cpu_ms else 0.0
    gpu_rate = len(texts) / (gpu_ms / 1000) if gpu_ms else 0.0
    speedup = gpu_rate / cpu_rate if cpu_rate else 0.0
    compatible = dimensions_match and min(cosines) >= 0.9995 and sum(cosines) / len(cosines) >= 0.9999
    qualified = compatible and top1 >= 0.99 and recall10 >= 0.99 and speedup >= 1.5
    report = {
        "qualified": qualified, "compatible": compatible, "device": "intel_gpu", "backend": gpu.backend,
        "model_id": gpu.model_id, "dimension": gpu.dimension, "texts": len(texts), "batch_size": batch_size,
        "device_name": gpu.device_name, "openvino_version": gpu.runtime_version,
        "cpu_initialization_ms": cpu_init_ms, "gpu_initialization_ms": gpu_init_ms,
        "cpu_embedding_ms": cpu_ms, "gpu_embedding_ms": gpu_ms,
        "cpu_chunks_s": cpu_rate, "gpu_chunks_s": gpu_rate, "speedup": speedup,
        "minimum_cosine": min(cosines), "mean_cosine": sum(cosines) / len(cosines),
        "maximum_component_delta": max(max_deltas),
        "mean_component_delta": sum(mean_deltas) / len(mean_deltas),
        "top_1_agreement": top1, "recall_at_5": recall5, "recall_at_10": recall10,
    }
    atomic_write_json(root / "qualification.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-db")
    parser.add_argument("--count", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64, choices=(16, 32, 64, 128, 256))
    parser.add_argument("--model", default="balanced")
    parser.add_argument("--runtime-dir")
    args = parser.parse_args(argv)
    print(json.dumps(qualify(sample_db=args.sample_db, count=args.count, batch_size=args.batch_size,
                             model=args.model, runtime_dir=args.runtime_dir), indent=2))


if __name__ == "__main__":
    main()
