"""PR46 factual evidence precision and raw/presentation separation."""
from axp_client.rag.retrieval import (
    classify_passages,
    classify_query_evidence_intent,
    detect_scalar_evidence,
    document_identity_strength,
    rank_documents,
    retrieve_rag_candidates,
)


def passage(text, filename="HEPTANE 99% FDS.pdf", **extra):
    return {"document_id": 1, "chunk_id": 1, "chunk_no": 20, "snippet": text,
            "filename": filename, "passage_score": .5, "vector_similarity": .7, **extra}


def test_scalar_intent_separates_identity_and_target_conservatively():
    intent = classify_query_evidence_intent("What is the density of n-Heptane 99%?")
    assert intent.kind == "scalar_fact" and intent.target_terms == {"density"}
    assert {"heptane", "99%"} <= intent.identity_terms
    assert classify_query_evidence_intent("packaging options for ammonia").kind == "general_semantic"
    assert classify_query_evidence_intent("handling of ammonia").kind == "general_semantic"


def test_direct_density_shapes_and_title_only_false_positive():
    intent = classify_query_evidence_intent("Heptane 99% density")
    rows = [passage("Density at 20°C: 0.68 g/cm³"),
            passage("Density measurements were performed during development.", chunk_id=2)]
    classify_passages(rows, intent)
    assert rows[0]["evidence_tier"] == "DIRECT_ANSWER"
    assert rows[1]["evidence_tier"] == "TOPICAL_ONLY"
    assert rows[0]["answer_shape_detected"] is True


def test_same_row_and_sentence_are_direct_answer_shapes():
    for text in ("Density | 0.684 g/cm³", "The density at 20°C is 0.684 g/cm³."):
        assert detect_scalar_evidence(text, {"density"})["answer_shape_detected"]


def test_conditions_and_administrative_numbers_are_not_density_answers():
    assert not detect_scalar_evidence("Density at 20°C", {"density"})["answer_shape_detected"]
    assert not detect_scalar_evidence("Density\nBatch 12345\nDate 2026", {"density"})["answer_shape_detected"]
    distant = "Density" + (" unrelated" * 40) + " 0.684 g/cm³"
    assert not detect_scalar_evidence(distant, {"density"})["answer_shape_detected"]


def test_neighbor_split_upgrades_to_direct_answer():
    intent = classify_query_evidence_intent("Heptane 99% density")
    rows = [passage("Density at 20°C"), passage("0.684 g/cm³", chunk_id=2, chunk_no=21)]
    classify_passages(rows, intent)
    assert rows[0]["evidence_tier"] == "DIRECT_ANSWER"


def test_mtbe_sds_has_exact_identity_and_bounded_bonus_wins():
    intent = classify_query_evidence_intent("density of MTBE")
    assert document_identity_strength("MSDS MTBE SIMFEX.pdf", intent.identity_terms) == "exact"
    direct = passage("Density: 0.74 g/cm³", "MSDS MTBE SIMFEX.pdf", document_id=1)
    generic = [passage("MTBE density measurements report", "Project report.pdf", document_id=2,
                       chunk_id=index, passage_score=.52) for index in range(2, 5)]
    classify_passages([direct, *generic], intent)
    assert rank_documents([direct, *generic], intent=intent)[0]["document_id"] == 1


def test_rag_uses_explicit_raw_boundary_not_presentation():
    calls = []
    def presentation(*args, **kwargs):
        raise AssertionError("presentation must be bypassed")
    def raw(*args, **kwargs):
        calls.append(True)
        return {"results": []}
    presentation.raw_search = raw
    class Connection:
        pass
    result = retrieve_rag_candidates(Connection(), None, "heptane", search_fn=presentation)
    assert calls and result.candidates == []
