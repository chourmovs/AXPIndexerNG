"""Shared, inspectable document metadata identity matching.

This module deliberately returns ranking dimensions rather than blending metadata
authority into passage relevance.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from .fts import TOKEN_RE
from .identifiers import extract_identifiers

QUERY_STOPWORDS = {
    "avec", "dans", "des", "est", "les", "par", "pour", "que", "quel", "quelle", "quels",
    "quelles", "qui", "sont", "sur", "trouve", "donne", "une", "un", "de", "du", "la", "le",
    "and", "for", "from", "the", "this", "with", "what", "which", "where", "is", "are", "of",
    "a", "an", "in", "at", "give", "find", "show", "open", "ouvre", "ouvrir", "cherche", "montre",
    "resume", "resumer", "synthese", "document", "rapport", "report", "procedure", "dit", "moi",
    "fais", "peux", "peut", "please",
}

# A lone format/category noun is not a safe document identity.  Multi-term names
# containing these words remain eligible when another distinctive term is present.
GENERIC_IDENTITY_TERMS = {
    "document", "report", "rapport", "summary", "resume", "sequence", "procedure", "process",
    "development", "final", "strategy", "validation", "specification", "spec",
}


def fold_search_text(value):
    decomposed = unicodedata.normalize("NFKD", str(value or ""))
    return "".join(character for character in decomposed
                   if not unicodedata.combining(character)).casefold()


def _canonical_term(value):
    value = fold_search_text(value)
    # Collection labels are commonly plural while natural-language queries are singular.
    return value[:-1] if len(value) > 4 and value.endswith("s") else value


def _ordered_terms(value, *, meaningful=True):
    terms = []
    for match in TOKEN_RE.finditer(str(value or "")):
        token = fold_search_text(match.group(0))
        parts = [part for part in re.split(r"[-._/\\]+", token) if part]
        for part in parts:
            canonical = _canonical_term(part)
            compound_component = len(parts) > 1 and bool(re.search(r"\d", token) or len(canonical) >= 1)
            if not meaningful or ((len(canonical) >= 3 or compound_component)
                                  and canonical not in QUERY_STOPWORDS):
                terms.append(canonical)
    return terms


def _meaningful_terms(value):
    """Return the historical search term set used by passage/query logic.

    Short components are useful when comparing an explicit compound filename as
    an ordered phrase (for example ``PROJECT-X``), but must not leak into the
    general query-term API.  In particular, the ``n`` in ``n-Heptane`` is not a
    standalone factual search term.
    """
    terms = set()
    for match in TOKEN_RE.finditer(str(value or "")):
        token = fold_search_text(match.group(0))
        variants = (token, *re.split(r"[-._/\\]+", token))
        terms.update(_canonical_term(part) for part in variants
                     if len(part) >= 3 and _canonical_term(part) not in QUERY_STOPWORDS)
    return terms


def _is_phrase(query_terms, metadata_terms):
    if not metadata_terms or len(metadata_terms) > len(query_terms):
        return False
    width = len(metadata_terms)
    return any(query_terms[index:index + width] == metadata_terms
               for index in range(len(query_terms) - width + 1))


@dataclass(frozen=True)
class DocumentIdentityMatch:
    exact_identifier_match: bool
    filename_identity_match: bool
    title_identity_match: bool
    filename_coverage: float
    title_coverage: float
    source_label_coverage: float
    source_path_coverage: float
    metadata_identity_coverage: float
    collection_coverage: float
    identity_terms: frozenset[str]
    collection_terms: frozenset[str]
    priority: tuple[int, int, int, int]


def analyze_document_identity(query, *, filename="", title="", source_label="", source_path=""):
    """Analyze explicit identity against one bounded document candidate."""
    query_ordered = _ordered_terms(query)
    query_terms = set(query_ordered)
    label_terms = _meaningful_terms(source_label)
    path_terms = _meaningful_terms(source_path)
    collection_terms = query_terms & (label_terms | path_terms)
    identity_terms = query_terms - collection_terms

    filename_terms = _ordered_terms(Path(str(filename or "")).stem)
    title_terms = _ordered_terms(title)
    distinctive_filename = set(filename_terms) - GENERIC_IDENTITY_TERMS
    distinctive_title = set(title_terms) - GENERIC_IDENTITY_TERMS
    filename_match = bool(distinctive_filename and _is_phrase(query_ordered, filename_terms))
    title_match = bool(distinctive_title and _is_phrase(query_ordered, title_terms))

    def coverage(terms, wanted=identity_terms):
        return len(wanted & set(terms)) / len(wanted) if wanted else 0.0

    filename_coverage = coverage(filename_terms)
    title_coverage = coverage(title_terms)
    metadata_coverage = coverage(set(filename_terms) | set(title_terms))
    source_label_coverage = (len(collection_terms & label_terms) / len(collection_terms)
                             if collection_terms else 0.0)
    source_path_coverage = (len(collection_terms & path_terms) / len(collection_terms)
                            if collection_terms else 0.0)
    collection_coverage = max(source_label_coverage, source_path_coverage)
    query_ids = {value for value, _ in extract_identifiers(query)}
    metadata_ids = {value for value, _ in extract_identifiers(f"{filename} {title}")}
    exact_identifier = bool(query_ids & metadata_ids)
    complete = bool(identity_terms and metadata_coverage == 1.0)
    partial = bool(metadata_coverage)
    priority = (int(exact_identifier), int(filename_match), int(title_match), int(complete or partial))
    return DocumentIdentityMatch(
        exact_identifier, filename_match, title_match, filename_coverage, title_coverage,
        source_label_coverage, source_path_coverage, metadata_coverage, collection_coverage,
        frozenset(identity_terms), frozenset(collection_terms), priority,
    )


def identity_diagnostics(match):
    return {
        "filename_identity_match": match.filename_identity_match,
        "title_identity_match": match.title_identity_match,
        "metadata_identity_coverage": match.metadata_identity_coverage,
        "collection_coverage": match.collection_coverage,
        "identity_terms": sorted(match.identity_terms),
        "collection_terms": sorted(match.collection_terms),
        "document_identity_priority": match.priority,
    }
