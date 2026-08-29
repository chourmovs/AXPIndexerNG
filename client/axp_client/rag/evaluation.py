from __future__ import annotations

import json
import statistics
import time
from dataclasses import asdict
from pathlib import Path

from .answerability import AnswerabilityConfig, decide_answerability


def confusion(outcomes):
    tp = sum(x["expected_answerable"] and x["actual_answerable"] for x in outcomes)
    tn = sum(not x["expected_answerable"] and not x["actual_answerable"] for x in outcomes)
    fp = sum(not x["expected_answerable"] and x["actual_answerable"] for x in outcomes)
    fn = sum(x["expected_answerable"] and not x["actual_answerable"] for x in outcomes)
    positives, negatives = tp + fn, tn + fp
    return {"correct_accepts": tp, "correct_refusals": tn, "false_accepts": fp, "false_refusals": fn,
            "precision": tp / (tp + fp) if tp + fp else 0, "recall": tp / positives if positives else 0,
            "false_accept_rate": fp / negatives if negatives else 0,
            "false_refusal_rate": fn / positives if positives else 0}


def threshold_sweep(cases_with_hits, configs=None):
    configs = configs or [AnswerabilityConfig(strong_vector_similarity=strong, support_vector_similarity=support)
                          for strong in (0.50, 0.55, 0.60) for support in (0.40, 0.45, 0.50) if support < strong]
    values = []
    for config in configs:
        outcomes = [{"expected_answerable": case["expected_answerable"],
                     "actual_answerable": decide_answerability(case["hits"], config).answerable}
                    for case in cases_with_hits]
        values.append({"config": asdict(config), "metrics": confusion(outcomes)})
    return sorted(values, key=lambda x: (x["metrics"]["false_accept_rate"], -x["metrics"]["recall"],
                                         x["config"]["strong_vector_similarity"]))


def evaluate(cases, runner, *, mode="gate-only"):
    outcomes, retrieval_times, generation_times = [], [], []
    for case in cases:
        started = time.perf_counter()
        value = runner(case["question"], mode)
        elapsed = (time.perf_counter() - started) * 1000
        timings = value.get("timings", {})
        retrieval_times.append(timings.get("retrieval_ms", elapsed))
        if timings.get("generation_ms") is not None:
            generation_times.append(timings["generation_ms"])
        answer = value.get("answer") or ""
        terms = case.get("expected_terms", [])
        outcomes.append({"id": case["id"], "question": case["question"],
                         "expected_answerable": bool(case["expected_answerable"]),
                         "actual_answerable": bool(value.get("answerable")),
                         "reason": value.get("decision", {}).get("reason"),
                         "citation_valid": value.get("status") == "answered" if mode == "full" else None,
                         "expected_terms_present": all(term.casefold() in answer.casefold() for term in terms) if terms else None})
    metrics = confusion(outcomes)
    latency = {"p50_retrieval_ms": statistics.median(retrieval_times) if retrieval_times else None,
               "p50_generation_ms": statistics.median(generation_times) if generation_times else None,
               "p95_generation_ms": sorted(generation_times)[max(0, int(len(generation_times) * .95) - 1)]
               if generation_times else None}
    return {"timestamp_ms": int(time.time() * 1000), "mode": mode, "cases": len(cases),
            "answerability_gate": metrics, "latency": latency, "outcomes": outcomes}


def load_cases(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, list) or any(not all(k in case for k in ("id", "question", "expected_answerable")) for case in value):
        raise ValueError("invalid RAG evaluation cases")
    return value


def format_summary(result):
    m = result["answerability_gate"]
    return (f"RAG evaluation\n\nCases: {result['cases']}\n\nAnswerability gate\n------------------\n"
            f"Correct accepts {m['correct_accepts']:7d}\nCorrect refusals {m['correct_refusals']:6d}\n"
            f"False accepts {m['false_accepts']:8d}\nFalse refusals {m['false_refusals']:7d}\n\n"
            f"Precision {m['precision']:15.1%}\nRecall {m['recall']:18.1%}\n"
            f"False accept rate {m['false_accept_rate']:7.1%}\nFalse refusal rate {m['false_refusal_rate']:6.1%}")
