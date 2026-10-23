"""Vector benchmark metrics; no external ANN dependency at import time."""
from __future__ import annotations

import math


def select_by_position(position, total, selected):
    """Deterministic spread across all rowids; position is one-based."""
    if total <= 0 or not (0 <= selected <= total) or not (1 <= position <= total):
        raise ValueError("invalid deterministic corpus sampling parameters")
    return position * selected // total > (position - 1) * selected // total


def recall_at_k(expected, actual, k):
    if k < 1:
        raise ValueError("k must be positive")
    truth = set(expected[:k])
    return len(truth.intersection(actual[:k])) / len(truth) if truth else 1.0


def exact_cosine_topk(normalized_vectors, keys, query, k):
    """Exhaustive FP32 cosine reference over exactly the sampled vector set."""
    import numpy as np

    q = np.asarray(query, dtype=np.float32)
    norm = float(np.linalg.norm(q))
    if not math.isfinite(norm) or norm <= 1e-12:
        raise ValueError("invalid zero/NaN query embedding")
    if k < 1 or len(keys) != len(normalized_vectors):
        raise ValueError("invalid k or mismatch keys/matrix")
    scores = normalized_vectors @ (q / norm)
    count = min(k, len(scores))
    if not count:
        return []
    top = np.argpartition(-scores, count - 1)[:count]
    # Use chunk rowids as deterministic tiebreakers. Approximate boundary ties
    # can still make recall ambiguous; report the metric with that limitation.
    return [int(keys[i]) for i in sorted(top, key=lambda i: (-float(scores[i]), int(keys[i])))]
