from axp_core.database import connect
from axp_core.sources import add_source, disable_source
from axp_daemon import indexer
from axp_daemon.indexer import scan_all, scan_source
from conftest import FakeEmbedder


def test_offline_source_never_deletes_existing_documents(tmp_path):
    root = tmp_path / "network"
    root.mkdir()
    (root / "one.txt").write_text("reactor pressure")
    con = connect(tmp_path / "offline.db", dimension=3)
    source = add_source(con, root)
    scan_source(con, source["id"], FakeEmbedder())
    root.rename(tmp_path / "disconnected")
    result = scan_source(con, source["id"], FakeEmbedder())
    assert result["status"] == "offline" and not result["scan_complete"]
    assert con.execute("SELECT count(*) FROM documents WHERE source_id=?", (source["id"],)).fetchone()[0] == 1


def test_partial_enumeration_never_mass_deletes(tmp_path, monkeypatch):
    root = tmp_path / "partial"
    root.mkdir()
    first = root / "first.txt"
    second = root / "second.txt"
    first.write_text("reactor")
    second.write_text("storage")
    con = connect(tmp_path / "partial.db", dimension=3)
    source = add_source(con, root)
    scan_source(con, source["id"], FakeEmbedder())

    class Partial:
        complete = False

        def __init__(self):
            self.errors = ["access denied in a subdirectory"]

        def __iter__(self):
            yield first

    monkeypatch.setattr(indexer, "discover", lambda *args, **kwargs: Partial())
    result = scan_source(con, source["id"], FakeEmbedder())
    assert not result["scan_complete"]
    assert con.execute("SELECT count(*) FROM documents WHERE source_id=?", (source["id"],)).fetchone()[0] == 2


def test_multi_source_scan_and_deletion_are_isolated(tmp_path):
    a_root, b_root = tmp_path / "a", tmp_path / "b"
    a_root.mkdir(); b_root.mkdir()
    a_file, b_file = a_root / "a.txt", b_root / "b.txt"
    a_file.write_text("reactor"); b_file.write_text("storage")
    con = connect(tmp_path / "multi.db", dimension=3)
    a, b = add_source(con, a_root), add_source(con, b_root)
    embedder = FakeEmbedder()
    scan_all(con, embedder)
    a_file.unlink()
    scan_source(con, a["id"], embedder)
    assert con.execute("SELECT count(*) FROM documents WHERE source_id=?", (a["id"],)).fetchone()[0] == 0
    assert con.execute("SELECT count(*) FROM documents WHERE source_id=?", (b["id"],)).fetchone()[0] == 1
    disable_source(con, a["id"])
    b_file.write_text("storage changed")
    scan_all(con, embedder)
    assert con.execute("SELECT sha256 FROM documents WHERE source_id=?", (b["id"],)).fetchone()[0]


def test_disabling_source_during_scan_never_reconciles_unseen_files(tmp_path):
    root = tmp_path / "disable"
    root.mkdir()
    (root / "a.txt").write_text("reactor")
    (root / "b.txt").write_text("storage")
    con = connect(tmp_path / "disable.db", dimension=3)
    source = add_source(con, root)
    embedder = FakeEmbedder()
    scan_source(con, source["id"], embedder)

    class DisableAfterFirst:
        done = False

        def current_file(self, *_):
            if not self.done:
                disable_source(con, source["id"])
                self.done = True

    result = scan_source(con, source["id"], embedder, control=DisableAfterFirst())
    assert not result["scan_complete"] and result["status"] == "disabled"
    assert con.execute("SELECT count(*) FROM documents WHERE source_id=?", (source["id"],)).fetchone()[0] == 2
