"""PR-A: reliable percentile aggregation for cold/warm Search runs."""
import pytest

from axp_core.performance import nearest_rank, summarize_latency_rows


def test_nearest_rank_keeps_small_sample_semantics():
    assert nearest_rank([8.0], 95) == 8.0
    assert nearest_rank([1, 2, 3, 4, 5], 50) == 3.0
    assert nearest_rank([1, 2, 3, 4, 5], 95) == 5.0
    assert nearest_rank([5, 4, 3, 2, 1], 99) == 5.0
    with pytest.raises(ValueError):
        nearest_rank([], 50)
    with pytest.raises(ValueError):
        nearest_rank([1], 101)


def test_benchmark_summary_does_not_mix_phases_or_queries():
    rows = [
        {"query": "MTBE", "cold/warm": "cold_reader", "total_ms": 12},
        {"query": "MTBE", "cold/warm": "warm_reader", "total_ms": 4},
        {"query": "MTBE", "cold/warm": "warm_reader", "total_ms": 5},
        {"query": "DOMINO", "cold/warm": "warm_reader", "total_ms": 8},
    ]
    groups = summarize_latency_rows(rows)
    assert len(groups) == 3
    warm = next(x for x in groups if x["query"] == "MTBE" and x["phase"] == "warm_reader")
    assert warm["metrics"]["total_ms"] == {"n": 2, "min": 4.0, "p50": 4.0,
                                          "p95": 5.0, "p99": 5.0, "max": 5.0}
