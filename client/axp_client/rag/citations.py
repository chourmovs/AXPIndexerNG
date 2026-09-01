import re

CITATION_RE = re.compile(r"\[S(\d+)\]")


def validate_citations(answer, supplied_ids):
    reason, cited = classify_citations(answer, supplied_ids)
    return reason == "valid", cited


def classify_citations(answer, supplied_ids):
    cited = {f"S{number}" for number in CITATION_RE.findall(answer or "")}
    prose = CITATION_RE.sub("", answer or "").strip()
    if not cited:
        return "missing_citation", cited
    if not cited <= set(supplied_ids):
        return "unknown_citation", cited
    if not prose:
        return "citation_only_no_prose", cited
    return "valid", cited
