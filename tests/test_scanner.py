from axp_core.database import connect
from axp_daemon.indexer import scan
from conftest import FakeEmbedder


def test_multi_root(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    (a / "a.txt").write_text("alpha")
    (b / "b.txt").write_text("beta")
    c = connect(tmp_path / "x.db", dimension=3)
    scan(c, a, FakeEmbedder())
    scan(c, b, FakeEmbedder())
    (a / "a.txt").unlink()
    scan(c, a, FakeEmbedder())
    assert [r[0] for r in c.execute("select path from documents")] == [str(b / "b.txt")]
