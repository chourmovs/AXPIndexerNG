from pathlib import Path

import pytest

from axp_client.rag.answerability import DecisionReason, decide_answerability, is_supporting_evidence
from axp_client.rag.retrieval import rank_documents
from axp_client.rag.service import select_supporting_documents
from axp_core.hybrid import SearchConfig, _meaningful_terms, _relevance
from axp_core.identifiers import extract_identifiers

from test_rag import service


def evidence(document_id, score, *, chunk_id=1, vector=.58, lexical=.45, **extra):
    return {
        "document_id": document_id, "chunk_id": chunk_id, "chunk_no": chunk_id - 1,
        "filename": f"doc{document_id}.txt", "title": f"Document {document_id}",
        "evidence_score": score, "relevance_score": score, "vector_similarity": vector,
        "content_lexical_coverage": lexical, **extra,
    }


def test_multiple_support_below_point_55_reaches_document_selection():
    rows = [evidence(1, .48, chunk_id=1), evidence(1, .46, chunk_id=2, vector=.56)]
    decision = decide_answerability(rows)
    supporting, selected = select_supporting_documents(rows)
    assert decision.answerable and decision.reason == DecisionReason.MULTIPLE_SUPPORT
    assert len(supporting) == 2 and selected[0]["best_evidence_score"] == pytest.approx(.48)


def test_weak_positive_document_is_not_selected():
    rows = [evidence(1, .30, vector=.30, lexical=.10)]
    supporting, selected = select_supporting_documents(rows)
    assert not supporting and not selected


def test_exact_support_and_identifier_semantics_are_preserved():
    exact = evidence(1, .40, vector=.46, lexical=0, exact_content_identifier_match=True)
    assert is_supporting_evidence(exact)
    assert decide_answerability([exact]).reason == DecisionReason.EXACT_SUPPORTED
    assert {value for value, _ in extract_identifiers("CZL-0181-106 R042500")} == {"CZL0181106", "R042500"}


def test_dense_supporting_document_ranking_is_preserved():
    dense = [evidence(1, score, chunk_id=index) for index, score in enumerate((.48, .47, .45), 1)]
    isolated = [evidence(2, score, chunk_id=index + 3) for index, score in enumerate((.52, .20, .15), 1)]
    _, selected = select_supporting_documents(dense + isolated)
    assert rank_documents(dense + isolated)[0]["document_id"] == 1
    assert selected[0]["document_id"] == 1


@pytest.mark.parametrize("variant", ["n-heptane", "n heptane", "n_heptane", "n.heptane", "n/heptane"])
def test_title_separator_normalization_without_reindex(variant):
    query_terms = _meaningful_terms(f"density of {variant} 99%")
    matching = {"snippet": "density value", "identifiers": "", "title": "n heptane 99",
                "filename": "n-heptane 99.pdf", "heading": "", "vector_distance": .42, "lexical_rank": 1,
                "exact_identifier_match": False, "exact_phrase_match": False, "exact_filename_match": False,
                "exact_priority": 0}
    unrelated = {**matching, "title": "529166 Layout v1", "filename": "layout.pdf"}
    _relevance(matching, query_terms, SearchConfig(min_lexical_coverage=.1))
    _relevance(unrelated, query_terms, SearchConfig(min_lexical_coverage=.1))
    assert matching["title_coverage"] > 0 and matching["filename_coverage"] > 0
    assert matching["evidence_score"] > unrelated["evidence_score"]


def test_scalar_service_rejects_24_topical_sub_point_55_hits(tmp_path):
    documents = tuple((document_id, "content") for document_id in range(1, 9))
    rows = []
    for document_id in range(1, 9):
        for offset in range(3):
            chunk_id = (document_id - 1) * 3 + offset + 1
            rows.append(evidence(document_id, .48 - offset * .02, chunk_id=chunk_id,
                                 vector=.58 - offset * .01, lexical=.45))
    rag, backend = service(tmp_path, rows, documents=documents)
    response = rag.ask("What is the density of n-Heptane 99%?")
    assert response["status"] == "insufficient_evidence"
    assert not backend.calls


def test_search_more_status_and_depth_contract_is_explicit():
    source = (Path(__file__).parents[1] / "client/axp_client/web/ask.js").read_text(encoding="utf-8")
    assert "['answered','insufficient_evidence','ungrounded_generation','local_generation_skipped_latency_budget'].includes(response.status)" in source
    assert "response.context?.search_depth !== 1" in source
    assert "await askStream(question," in source and "},1);" in source
    assert "const question=article?.dataset.question" in source
    assert "article.replaceChildren(heading)" in source  # replacement happens only after a successful final response
