from dataclasses import dataclass
from enum import StrEnum


class DecisionReason(StrEnum):
    NO_CONTENT_EVIDENCE = "no_content_evidence"
    STRONG_EVIDENCE = "strong_evidence"
    MULTIPLE_SUPPORT = "multiple_support"
    EXACT_SUPPORTED = "exact_supported"
    WEAK_RETRIEVAL = "weak_retrieval"


@dataclass(frozen=True)
class AnswerabilityConfig:
    strong_relevance: float = 0.60
    support_relevance: float = 0.45
    multi_support_best: float = 0.50
    exact_supported: float = 0.45
    meaningful_lexical_support: float = 0.25


@dataclass(frozen=True)
class AnswerabilityDecision:
    answerable: bool
    reason: DecisionReason
    best_relevance: float
    supporting_chunks: int
    content_documents: int

    def public(self):
        return {"reason": self.reason.value, "best_relevance": self.best_relevance}


def decide_answerability(results, config=None):
    config = config or AnswerabilityConfig()
    if not results:
        return AnswerabilityDecision(False, DecisionReason.NO_CONTENT_EVIDENCE, 0.0, 0, 0)
    scores = [float(row.get("relevance_score") or 0) for row in results]
    best = max(scores)
    supporting = sum(score >= config.support_relevance for score in scores)
    documents = len({row["document_id"] for row in results})
    if best >= config.strong_relevance:
        reason, accepted = DecisionReason.STRONG_EVIDENCE, True
    elif best >= config.multi_support_best and supporting >= 2:
        reason, accepted = DecisionReason.MULTIPLE_SUPPORT, True
    elif any(
        float(row.get("relevance_score") or 0) >= config.exact_supported
        and float(row.get("lexical_coverage") or 0) >= config.meaningful_lexical_support
        and any(row.get(key) for key in ("exact_identifier_match", "exact_filename_match", "exact_phrase_match"))
        for row in results
    ):
        reason, accepted = DecisionReason.EXACT_SUPPORTED, True
    else:
        reason, accepted = DecisionReason.WEAK_RETRIEVAL, False
    return AnswerabilityDecision(accepted, reason, best, supporting, documents)
