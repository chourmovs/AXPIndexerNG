import math
import statistics


def evaluate(cases, search):
    reciprocal, recall5, recall10, latencies = [], [], [], []
    for case in cases:
        rows, elapsed_ms = search(case["query"])
        ids = [row["document_id"] for row in rows]
        relevant = set(case["relevant"])
        recall5.append(bool(relevant.intersection(ids[:5])))
        recall10.append(bool(relevant.intersection(ids[:10])))
        ranks = [i + 1 for i, value in enumerate(ids[:10]) if value in relevant]
        reciprocal.append(1 / min(ranks) if ranks else 0)
        latencies.append(elapsed_ms)
    ordered = sorted(latencies)
    p95 = ordered[min(len(ordered) - 1, math.ceil(len(ordered) * 0.95) - 1)] if ordered else 0
    return {"recall@5": statistics.mean(recall5), "recall@10": statistics.mean(recall10), "mrr@10": statistics.mean(reciprocal), "average_ms": statistics.mean(latencies), "p95_ms": p95}
