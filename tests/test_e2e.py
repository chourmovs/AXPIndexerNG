from axp_core.database import connect
from axp_core.hybrid import search
from axp_daemon.indexer import scan
from conftest import FakeEmbedder


def test_e2e_delete(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    r = root / "reactor.txt"
    r.write_text("The pressure control valve regulates reactor pressure.")
    (root / "warehouse.txt").write_text("Storage and logistics inventory.")
    e = FakeEmbedder()
    c = connect(tmp_path / "x.db", dimension=e.dimension)
    scan(c, root, e)
    assert search(c, "reactor pressure", e.embed_query("reactor pressure"))[0]["path"].endswith("reactor.txt")
    reactor_document_id = c.execute("SELECT id FROM documents WHERE path LIKE ?", ("%reactor.txt",)).fetchone()[0]
    reactor_chunk_ids = [
        row[0] for row in c.execute("SELECT id FROM chunks WHERE document_id = ?", (reactor_document_id,))
    ]
    assert reactor_chunk_ids

    r.unlink()
    scan(c, root, e)

    assert not c.execute("SELECT 1 FROM documents WHERE id = ?", (reactor_document_id,)).fetchall()
    assert not c.execute("SELECT 1 FROM chunks WHERE document_id = ?", (reactor_document_id,)).fetchall()
    assert not c.execute(
        "SELECT 1 FROM chunks_fts WHERE rowid IN ({})".format(",".join("?" for _ in reactor_chunk_ids)),
        reactor_chunk_ids,
    ).fetchall()
    assert not c.execute(
        "SELECT 1 FROM chunk_vectors WHERE rowid IN ({})".format(",".join("?" for _ in reactor_chunk_ids)),
        reactor_chunk_ids,
    ).fetchall()

    remaining_results = search(c, "reactor", e.embed_query("reactor"))
    assert all(not result["path"].endswith("reactor.txt") for result in remaining_results)
