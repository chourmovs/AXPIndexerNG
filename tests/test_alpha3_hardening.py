from pathlib import Path
from types import SimpleNamespace

import pytest
from axp_core.database import connect
from axp_core.sources import add_source
from axp_daemon import service
from axp_daemon.indexer import scan_source
from axp_daemon.scanner import discover
from conftest import FakeEmbedder


def test_office_temporary_files_are_not_discovered(tmp_path):
    (tmp_path / "report.docx").write_bytes(b"document")
    (tmp_path / "~$report.docx").write_bytes(b"office lock")
    (tmp_path / ".~lock.report.docx#").write_bytes(b"libreoffice lock")
    assert [path.name for path in discover(tmp_path)] == ["report.docx"]


def test_embedding_batch_failure_isolates_one_document_and_continues(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    for name, text in (("a.txt", "reactor"), ("b.txt", "storage"),
                       ("bad.txt", "poison payload"), ("c.txt", "pressure")):
        (root / name).write_text(text, encoding="utf-8")
    con = connect(tmp_path / "catalog.db", dimension=3)
    source = add_source(con, root)

    class IsolatingEmbedder(FakeEmbedder):
        def embed_documents(self, texts):
            if any("poison" in text for text in texts):
                raise RuntimeError("synthetic backend failure")
            return super().embed_documents(texts)

    result = scan_source(con, source["id"], IsolatingEmbedder(), embedding_batch_size=100)
    indexed = {Path(row[0]).name for row in con.execute("SELECT path FROM documents")}
    assert result["scan_complete"] and result["status"] == "idle"
    assert result["files_failed"] == 1
    assert indexed == {"a.txt", "b.txt", "c.txt"}


class Publisher:
    def __init__(self):
        self.value = {}

    def update(self, **values):
        self.value.update(values)


class Control:
    stop = False

    def poll(self):
        pass


def test_missing_model_is_downloaded_then_reopened(monkeypatch, tmp_path):
    calls = []
    expected = SimpleNamespace(model_id="test/model")

    def fake_embedder(profile, cache_dir, local_only):
        calls.append((profile, cache_dir, local_only))
        if local_only:
            raise ValueError("cache missing")
        return expected

    monkeypatch.setattr(service, "Embedder", fake_embedder)
    publisher = Publisher()
    result = service._provision_embedder("balanced", tmp_path / "model-cache", True, 60, publisher, Control())
    assert result is expected
    assert [call[2] for call in calls] == [True, False]
    assert publisher.value["model_download_attempts"] == 1
    assert publisher.value["last_error"] is None


def test_missing_model_without_download_has_actionable_error(monkeypatch, tmp_path):
    monkeypatch.setattr(service, "Embedder", lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("missing")))
    with pytest.raises(RuntimeError, match="missing or incomplete"):
        service._provision_embedder("balanced", tmp_path / "model-cache", False, 60, Publisher(), Control())
