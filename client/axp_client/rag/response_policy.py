"""Deterministic visible-response policy for reasoning-enabled RAG models."""
from dataclasses import dataclass
import re


@dataclass(frozen=True)
class ResponsePlan:
    mode: str
    answer_tokens: int
    target_words: int
    instruction: str


_SUMMARY_RE = re.compile(
    r"\b(?:summari[sz]e|summary|overview|main properties|key properties|key points|"
    r"résume|résumer|résumé|synthèse|synthétise|principales propriétés|points principaux)\b",
    re.IGNORECASE,
)
_ANALYTICAL_RE = re.compile(
    r"\b(?:compare|comparison|differences?|versus|vs|why|explain|advantages?|disadvantages?|"
    r"comparaison|différences?|pourquoi|explique|expliquer|avantages|inconvénients)\b",
    re.IGNORECASE,
)
_QUESTION_RE = re.compile(
    r"^\s*(?:what|where|which|who|when|how|is|are|do|does|can|quelle?s?|quel(?:le)?s?|où|"
    r"qui|quand|comment|est-ce)\b",
    re.IGNORECASE,
)


def classify_response_plan(question, intent):
    """Map a conservative query shape and existing evidence intent to a response contract."""
    if intent.kind == "scalar_fact":
        return ResponsePlan("scalar_lookup", 224, 60, """Give the requested value directly.
Use at most 1–2 short sentences.
The first factual sentence MUST contain a valid supplied [Sx] citation.
Do not narrate how you found the answer.
Do not repeat or verify the same fact.
Stop once the requested fact has been answered.""")
    if _SUMMARY_RE.search(question):
        return ResponsePlan("summary", 320, 220, """Provide a compact synthesis.
Use concise bullets when several independent facts are requested.
Do not narrate source inspection.
Do not describe documents unless document identity itself matters.
Attach citations to the facts or bullet groups they support.
Do not add a conclusion that merely repeats the bullets.""")
    if _ANALYTICAL_RE.search(question):
        return ResponsePlan("analytical", 320, 220, """Answer the analytical question directly.
Use a compact structure.
Cite each material conclusion.
Do not add generic background unsupported by evidence.""")
    if question.rstrip().endswith("?") or _QUESTION_RE.search(question):
        return ResponsePlan("direct_lookup", 224, 120, """Answer immediately.
Use the shortest complete form appropriate to the requested information.
The first substantive sentence MUST contain a valid supplied citation.
Use bullets only when the answer naturally consists of multiple items or steps.
Do not narrate evidence inspection.
Stop once the question has been answered.""")
    return ResponsePlan("default", 256, 160, """Answer the request directly and concisely.
Attach supplied citations to the facts they support.
Do not narrate evidence inspection or add unsupported background.
Stop once the request has been answered.""")
