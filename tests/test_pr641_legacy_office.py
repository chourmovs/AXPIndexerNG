import pytest

from axp_daemon.extractors import doc, ppt, xls
from axp_daemon.extractors.legacy_office import OfficeExtractionError
from axp_daemon.scanner import is_supported_document


@pytest.mark.parametrize("name", ["PROJECT.doc", "PROJECT.docx", "PROJECT.xls", "PROJECT.xlsx",
                                         "PROJECT.ppt", "PROJECT.pptx"])
def test_mandatory_office_matrix_is_content_supported(name):
    assert is_supported_document(name)


def test_legacy_word_uses_parser_text_without_synthetic_metadata(monkeypatch, tmp_path):
    path = tmp_path / "PROJECT.doc"
    path.write_bytes(b"fixture adapter")
    monkeypatch.setattr("office_oxide.to_markdown", lambda value: "Reaction temperature 42 °C")
    assert doc.extract(path) == [("Reaction temperature 42 °C", None)]


def test_legacy_excel_preserves_sheet_order_and_values(monkeypatch, tmp_path):
    path = tmp_path / "PROJECT.xls"
    path.write_bytes(b"fixture adapter")
    monkeypatch.setattr("office_oxide.to_markdown", lambda value:
                        "# Sheet: Production\nA1 | Batch\n\n# Sheet: Results\nA1 | Passed")
    sections = xls.extract(path)
    assert [text.splitlines()[0] for text, _ in sections] == ["# Sheet: Production", "# Sheet: Results"]
    assert "Batch" in sections[0][0] and "Passed" in sections[1][0]


def test_legacy_powerpoint_preserves_slide_numbers(monkeypatch, tmp_path):
    path = tmp_path / "PROJECT.ppt"
    path.write_bytes(b"fixture adapter")
    monkeypatch.setattr("office_oxide.to_markdown", lambda value:
                        "# Slide 1\nOpening title\n\n# Slide 2\nConclusion")
    sections = ppt.extract(path)
    assert [page for _, page in sections] == [1, 2]
    assert "Opening title" in sections[0][0] and "Conclusion" in sections[1][0]


def test_empty_and_encrypted_are_explicit(monkeypatch, tmp_path):
    path = tmp_path / "bad.doc"
    path.write_bytes(b"fixture adapter")
    monkeypatch.setattr("office_oxide.to_markdown", lambda value: "")
    with pytest.raises(OfficeExtractionError, match="office_content_empty"):
        doc.extract(path)

    def encrypted(_value):
        raise ValueError("document is password encrypted")

    monkeypatch.setattr("office_oxide.to_markdown", encrypted)
    with pytest.raises(OfficeExtractionError, match="office_encrypted"):
        doc.extract(path)
