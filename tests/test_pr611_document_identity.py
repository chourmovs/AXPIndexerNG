from axp_core.document_identity import analyze_document_identity


def test_filename_stem_is_an_explicit_phrase_inside_query():
    match = analyze_document_identity("séquence PROJECT-X", filename="PROJECT-X.docx",
                                      source_label="SEQUENCES")
    assert match.filename_identity_match
    assert match.identity_terms == {"project", "x"}
    assert match.collection_terms == {"sequence"}
    assert match.metadata_identity_coverage == 1.0
    assert match.collection_coverage == 1.0


def test_identity_uses_boundaries_titles_and_ignores_extensions():
    assert not analyze_document_identity("latest testing report", filename="TEST.docx").filename_identity_match
    assert analyze_document_identity("séquence Project Alpha", filename="Project Alpha.pdf").filename_identity_match
    assert analyze_document_identity(
        "résume Process Development Final Report",
        title="Process Development Final Report",
    ).title_identity_match


def test_generic_single_word_filename_is_not_authoritative():
    assert not analyze_document_identity("show report", filename="Report.docx").filename_identity_match
