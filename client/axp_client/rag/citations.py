import re

CITATION_RE = re.compile(r"\[S(\d+)\]")


def validate_citations(answer, supplied_ids):
    cited = {f"S{number}" for number in CITATION_RE.findall(answer or "")}
    return bool(cited) and cited <= set(supplied_ids), cited
