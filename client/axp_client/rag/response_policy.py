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
        return ResponsePlan("scalar_lookup", 128, 40, """Give only the requested fact.
Prefer one sentence. Maximum two short sentences.
Include an allowed citation in the factual sentence.
Do not explain how you found it. Do not repeat it.
Do not add related properties unless explicitly requested.
Stop immediately after the answer.""")
    if _SUMMARY_RE.search(question):
        return ResponsePlan("summary", 288, 160, """Provide only the main requested properties.
Use a maximum 10 concise bullets. Target <= 160 words.
Omit \"no data available\" properties unless materially relevant.
Do not add a concluding paragraph that repeats the bullets (no repetitive conclusion).
Use allowed citations directly on the supported bullet or bullet group.""")
    if _ANALYTICAL_RE.search(question):
        return ResponsePlan("analytical", 320, 200, """Answer the analytical question directly.
Use <= 200 words by default and a compact comparison or analysis.
Cite each material conclusion using only allowed citations.
Do not add generic background unsupported by evidence.""")
    if question.rstrip().endswith("?") or _QUESTION_RE.search(question):
        return ResponsePlan("direct_lookup", 192, 100, """Answer the requested question directly.
Use <= 100 words whenever possible.
Use at most 6 bullets if the answer is naturally a list.
Do not add unrelated background. Use only allowed citations.
Stop when the requested information is complete.""")
    return ResponsePlan("default", 224, 140, """Answer the request directly and concisely.
Attach supplied citations to the facts they support.
Do not narrate evidence inspection or add unsupported background.
Stop once the request has been answered.""")


def canonicalize_scalar_response(answer, allowed_citation_ids):
    """Keep the first complete sentence containing an emitted allowed citation."""
    if not answer or answer.strip().upper().startswith("INSUFFICIENT_EVIDENCE"):
        return answer, False, None
    allowed = "|".join(re.escape(f"[{source_id}]") for source_id in allowed_citation_ids)
    if not allowed:
        return answer, False, None
    cited = re.search(allowed, answer)
    if not cited:
        return answer, False, None
    boundary = re.search(r"[.!?](?=\s|$)", answer[cited.end():])
    if not boundary:
        return answer, False, None
    cutoff = cited.end() + boundary.end()
    result = answer[:cutoff].strip()
    return (result, True, "scalar_first_allowed_cited_sentence") if result != answer.strip() else (answer, False, None)


def cleanup_truncated_tail(answer, allowed_citation_ids):
    """Remove only a minority, uncited dangling tail after complete cited lines."""
    if not answer or answer.strip().upper().startswith("INSUFFICIENT_EVIDENCE"):
        return answer, False
    allowed = "|".join(re.escape(f"[{source_id}]") for source_id in allowed_citation_ids)
    matches = list(re.finditer(allowed, answer)) if allowed else []
    if not matches:
        return answer, False
    end = matches[-1].end()
    # A cited unit is complete at its citation when followed by punctuation, a newline, or EOF.
    suffix = answer[end:]
    unit_end = re.match(r"[.!?]?\s*(?:\n|$)", suffix)
    if not unit_end:
        return answer, False
    cutoff = end + unit_end.end()
    tail = answer[cutoff:].strip()
    retained_units = [line for line in answer[:cutoff].splitlines() if line.strip()]
    trailing_units = [line for line in tail.splitlines() if line.strip()]
    if (not tail or re.search(allowed, tail) or
            len(trailing_units) >= len(retained_units)):
        return answer, False
    if re.search(r"[.!?][\]\"')]*$", tail):
        return answer, False
    return answer[:cutoff].strip(), True
