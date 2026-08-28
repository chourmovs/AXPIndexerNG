from datetime import datetime

from axp_core.database import connect
from axp_core.sources import coverage_percentages, get_source
from axp_daemon.indexer import scan
from axp_daemon.scanner import discover
from conftest import FakeEmbedder
import pytest


def test_mixed_coverage_and_metadata_only_is_searchable(tmp_path):
    root = tmp_path / "docs"
    (root / "nested").mkdir(parents=True)
    (root / "notes.txt").write_text("pressure procedure", encoding="utf-8")
    (root / "PID-R042500-rev3.dwg").write_bytes(b"\x00binary must not be parsed")
    (root / "mail.msg").write_bytes(b"message")
    (root / "~$temp.docx").write_bytes(b"temporary")
    (root / "nested" / "other.bin").write_bytes(b"other")

    con = connect(tmp_path / "catalog.db", dimension=3)
    result = scan(con, root, FakeEmbedder())

    assert result["files_seen"] == 5
    assert (result["files_content"], result["files_metadata"], result["files_ignored"], result["files_failed"]) == (1, 3, 1, 0)
    assert result["files_seen"] == sum(result[key] for key in
                                       ("files_content", "files_metadata", "files_ignored", "files_failed"))
    drawing = con.execute("SELECT * FROM documents WHERE filename='PID-R042500-rev3.dwg'").fetchone()
    assert drawing["ingestion_mode"] == "metadata"
    chunks = con.execute("SELECT text FROM chunks WHERE document_id=?", (drawing["id"],)).fetchall()
    assert len(chunks) == 1 and "PID-R042500-rev3.dwg" in chunks[0]["text"]
    assert con.execute("SELECT count(*) FROM chunks_fts WHERE chunks_fts MATCH 'R042500'").fetchone()[0] >= 1
    source = get_source(con, drawing["source_id"])
    assert source["last_seen_count"] == 5 and source["last_metadata_count"] == 3
    assert coverage_percentages(5, 1, 3) == (80.0, 20.0)


def test_metadata_file_is_not_hashed(tmp_path, monkeypatch):
    root = tmp_path / "docs"
    root.mkdir()
    large = root / "large.dwg"
    with large.open("wb") as stream:
        stream.truncate(128 * 1024 * 1024)
    monkeypatch.setattr("axp_daemon.indexer.sha256", lambda _path: (_ for _ in ()).throw(AssertionError("hashed")))
    con = connect(tmp_path / "catalog.db", dimension=3)
    assert scan(con, root, FakeEmbedder())["files_metadata"] == 1


def test_xlsx_and_csv_content_through_scan(tmp_path):
    Workbook = pytest.importorskip("openpyxl").Workbook
    root = tmp_path / "docs"
    root.mkdir()
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Scale-up"
    sheet.append(["Equipment", "Température", "Date"])
    sheet.append(["R042500", 75.2, datetime(2026, 8, 28, 10, 30)])
    hidden = workbook.create_sheet("Données cachées")
    hidden.sheet_state = "hidden"
    hidden.append(["Mélange", "réussi"])
    workbook.save(root / "study.xlsx")
    (root / "measurements.csv").write_bytes(
        'Équipement;Description;Pression\r\nR042600;"vanne; mélange";1,8\r\n\r\n'.encode("cp1252"))

    con = connect(tmp_path / "catalog.db", dimension=3)
    result = scan(con, root, FakeEmbedder())

    assert result["files_content"] == 2 and result["files_failed"] == 0
    text = "\n".join(row[0] for row in con.execute("SELECT text FROM chunks ORDER BY id"))
    for value in ("Scale-up", "R042500", "75.2", "2026-08-28", "Données cachées", "R042600", "vanne; mélange"):
        assert value in text


def test_discovery_counts_all_regular_files_and_ignores_symlink(tmp_path):
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (root / "unknown.xyz").write_bytes(b"x")
    (outside / "external.txt").write_text("outside")
    try:
        (root / "linked").symlink_to(outside, target_is_directory=True)
    except OSError:
        pass
    traversal = discover(root)
    assert list(traversal) == [root / "unknown.xyz"]
    assert traversal.discovered == 1


def test_metadata_document_can_transition_to_content(tmp_path, monkeypatch):
    root = tmp_path / "docs"
    root.mkdir()
    document = root / "future.legacy"
    document.write_text("future extracted content", encoding="utf-8")
    con = connect(tmp_path / "catalog.db", dimension=3)
    scan(con, root, FakeEmbedder())
    assert con.execute("SELECT ingestion_mode FROM documents").fetchone()[0] == "metadata"

    monkeypatch.setattr("axp_daemon.scanner.SUPPORTED", {".legacy"})
    monkeypatch.setitem(__import__("axp_daemon.extractors", fromlist=["EXTRACTORS"]).EXTRACTORS,
                        ".legacy", lambda path: [(path.read_text(encoding="utf-8"), None)])
    result = scan(con, root, FakeEmbedder())

    row = con.execute("SELECT id,ingestion_mode FROM documents").fetchone()
    assert row["ingestion_mode"] == "content"
    chunks = con.execute("SELECT text FROM chunks WHERE document_id=?", (row["id"],)).fetchall()
    assert len(chunks) == 1 and "future extracted content" in chunks[0]["text"]
    assert result["files_modified"] == 1
