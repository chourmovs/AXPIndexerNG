"""Conservative deterministic Skill matching."""
import re

from axp_core.hybrid import _meaningful_terms, fold_search_text
from axp_core.identifiers import extract_identifiers


def _identifier(value):
    return re.sub(r"[^a-z0-9]", "", fold_search_text(value))


def match_skill(question, skills):
    folded = fold_search_text(question)
    terms = _meaningful_terms(question)
    identifiers = {_identifier(value) for value, _kind in extract_identifiers(question)}
    candidates = []
    for skill in skills:
        if not skill.enabled:
            continue
        configured_ids = {_identifier(value) for value in skill.match.identifiers}
        identifier_count = len(identifiers & configured_ids)
        phrases = [fold_search_text(value) for value in skill.match.phrases]
        phrase_count = sum(1 for value in phrases if value and value in folded)
        keywords = {fold_search_text(value) for value in skill.match.keywords}
        keyword_count = len(terms & keywords)
        if not identifier_count and not phrase_count and keyword_count < 2:
            continue
        strength = (bool(identifier_count), bool(phrase_count), phrase_count, keyword_count, skill.priority)
        reason = "identifier" if identifier_count else "phrase" if phrase_count else "keywords"
        candidates.append((strength, skill.id, skill, reason))
    if not candidates:
        return None, None, {"match": "no_match"}
    candidates.sort(key=lambda item: (item[0], tuple(-ord(c) for c in item[1])), reverse=True)
    best = candidates[0]
    tied = [item for item in candidates if item[0] == best[0]]
    if len(tied) > 1:
        return None, None, {"match": "ambiguous_skill_match", "candidate_ids": sorted(x[1] for x in tied)}
    return best[2], best[3], {"match": "matched"}
