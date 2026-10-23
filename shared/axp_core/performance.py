"""Pure latency aggregation for read-only Search/RAG benchmarks.

Percentiles use the nearest-rank definition; groups with one sample remain
visible but must not be mistaken for statistically meaningful p95/p99.
"""
import math
from collections import defaultdict

LATENCY_KEYS = (
    "db_acquire_ms", "fts_ms", "vector_ms", "fusion_ms", "scoring_ms",
    "drilldown_ms", "total_ms",
)


def nearest_rank(values, percentile):
    """Return a deterministic nearest-rank percentile, including small samples."""
    if not values or not 0 <= percentile <= 100:
        raise ValueError("nonempty values and percentile in [0,100] required")
    ordered = sorted(float(value) for value in values)
    rank = max(1, math.ceil((percentile / 100) * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


def summarize_latency_rows(rows, *, latency_keys=LATENCY_KEYS):
    """Group by query / reader phase; do not mix cold and warm observations."""
    groups = defaultdict(list)
    for row in rows:
        groups[(str(row["query"]), str(row["cold/warm"]))].append(row)
    summary = []
    for (query, phase), samples in sorted(groups.items()):
        measures = {}
        for metric in latency_keys:
            values = [float(row[metric]) for row in samples if row.get(metric) is not None]
            if values:
                measures[metric] = {
                    "n": len(values), "min": min(values), "p50": nearest_rank(values, 50),
                    "p95": nearest_rank(values, 95), "p99": nearest_rank(values, 99),
                    "max": max(values),
                }
        summary.append({"query": query, "phase": phase, "n": len(samples), "metrics": measures})
    return summary
