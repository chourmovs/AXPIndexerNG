from axp_client.rag.retrieval import classify_query_evidence_intent, rank_documents


def _hit(document_id, filename, score, source="SEQUENCES"):
    return {
        "document_id": document_id, "chunk_id": document_id, "filename": filename, "title": "",
        "source_label": source, "source_path": f"K:/R&D/{source}", "passage_score": score,
        "evidence_score": score, "relevance_score": score, "title_coverage": 0,
    }


def test_no_skill_signature_explicit_filename_outranks_semantics():
    hits = [_hit(1, "ASSYS.docx", .99), _hit(2, "DOMINO.docx", .40),
            _hit(3, "PROJECT-X.docx", .80)]
    for query, expected in (("séquence ASSYS", "ASSYS.docx"),
                            ("séquence DOMINO", "DOMINO.docx"),
                            ("séquence PROJECT-X", "PROJECT-X.docx")):
        ranked = rank_documents(hits, intent=classify_query_evidence_intent(query), query=query)
        assert ranked[0]["filename"] == expected


def test_collection_is_secondary_tie_break_for_same_identity():
    query = "séquence DOMINO"
    hits = [_hit(1, "DOMINO.docx", .99, "REPORTS"), _hit(2, "DOMINO.docx", .40, "SEQUENCES")]
    ranked = rank_documents(hits, intent=classify_query_evidence_intent(query), query=query)
    assert ranked[0]["document_id"] == 2
    assert ranked[0]["collection_terms"] == ["sequence"]


def test_semantic_query_keeps_semantic_document_order():
    query = "problème de filtration au charbon"
    hits = [_hit(1, "ASSYS.docx", .40), _hit(2, "DOMINO.docx", .90)]
    ranked = rank_documents(hits, intent=classify_query_evidence_intent(query), query=query)
    assert ranked[0]["document_id"] == 2
