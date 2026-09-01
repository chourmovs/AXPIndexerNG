from dataclasses import dataclass
from enum import StrEnum


class DecisionReason(StrEnum):
    NO_CONTENT_EVIDENCE = "no_content_evidence"
    STRONG_EVIDENCE = "strong_evidence"
    MULTIPLE_SUPPORT = "multiple_support"
    EXACT_SUPPORTED = "exact_supported"
    TOPIC_WITHOUT_ANSWER = "topic_match_without_answer_support"
    WEAK_RETRIEVAL = "weak_retrieval"


@dataclass(frozen=True)
class AnswerabilityConfig:
    strong_vector_similarity: float = 0.55
    support_vector_similarity: float = 0.45
    strong_lexical_coverage: float = 0.50
    support_lexical_coverage: float = 0.25
    minimum_supporting_chunks: int = 2
    minimum_supporting_documents: int = 1


@dataclass(frozen=True)
class AnswerabilityDecision:
    answerable: bool
    confidence: str
    reason: DecisionReason
    signals: dict

    @property
    def best_relevance(self):
        return self.signals["best_vector_similarity"]

    @property
    def supporting_chunks(self):
        return self.signals["support_chunks"]

    @property
    def content_documents(self):
        return self.signals["documents"]

    def public(self):
        return {"answerable": self.answerable, "confidence": self.confidence,
                "reason": self.reason.value, "signals": self.signals}


def _vector(row):
    # Retrieval explain output supplies vector_similarity; old fixtures may only have relevance_score.
    return float(row.get("vector_similarity", row.get("relevance_score")) or 0)


def _lexical(row):
    return float(row.get("content_lexical_coverage", row.get("lexical_coverage")) or 0)


def is_supporting_evidence(row, config=None):
    """Return whether a content row satisfies the answerability support boundary."""
    config = config or AnswerabilityConfig()
    if row.get("evidence_tier") == "DIRECT_ANSWER":
        return True
    vector_supported = _vector(row) >= config.support_vector_similarity
    return bool(vector_supported and (
        _lexical(row) >= config.support_lexical_coverage
        or row.get("exact_content_identifier_match")
        or row.get("exact_content_phrase_match")
    ))


def decide_answerability(results, config=None):
    config = config or AnswerabilityConfig()
    if not results:
        signals = {"best_vector_similarity": 0.0, "best_lexical_coverage": 0.0, "strong_chunks": 0,
                   "support_chunks": 0, "documents": 0, "exact_content_matches": 0}
        return AnswerabilityDecision(False, "low", DecisionReason.NO_CONTENT_EVIDENCE, signals)
    strong = [r for r in results if _vector(r) >= config.strong_vector_similarity
              and _lexical(r) >= config.support_lexical_coverage]
    support = [r for r in results if is_supporting_evidence(r, config)]
    exact = [r for r in results if (r.get("exact_content_identifier_match") or r.get("exact_content_phrase_match"))
             and _vector(r) >= config.support_vector_similarity]
    direct = [r for r in results if r.get("evidence_tier") == "DIRECT_ANSWER"]
    signals = {"best_vector_similarity": max(map(_vector, results)),
               "best_lexical_coverage": max(_lexical(r) for r in results),
               "strong_chunks": len(strong), "support_chunks": len(support),
               "documents": len({r["document_id"] for r in results}), "exact_content_matches": len(exact)}
    if direct:
        return AnswerabilityDecision(True, "high", DecisionReason.STRONG_EVIDENCE, signals)
    if exact:
        return AnswerabilityDecision(True, "high", DecisionReason.EXACT_SUPPORTED, signals)
    if len(support) >= config.minimum_supporting_chunks:
        return AnswerabilityDecision(True, "medium", DecisionReason.MULTIPLE_SUPPORT, signals)
    if strong:
        return AnswerabilityDecision(True, "high", DecisionReason.STRONG_EVIDENCE, signals)
    reason = (DecisionReason.TOPIC_WITHOUT_ANSWER if signals["best_lexical_coverage"] >= config.strong_lexical_coverage
              else DecisionReason.WEAK_RETRIEVAL)
    return AnswerabilityDecision(False, "low", reason, signals)
