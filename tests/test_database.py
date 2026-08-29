import sqlite3

import pytest
from axp_core.database import capability_report, connect, open_catalog_reader


def test_database(tmp_path):
    c = connect(tmp_path / "x.db", dimension=3)
    assert c.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    assert capability_report(c)["fts5"]


def test_catalog_reader_is_lightweight_and_read_only(tmp_path):
    path = tmp_path / "x.db"
    with sqlite3.connect(path) as writer:
        writer.execute("CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT)")
        writer.execute("INSERT INTO metadata(key,value) VALUES('reader-test','ok')")
    with open_catalog_reader(path) as reader:
        assert reader.execute("PRAGMA query_only").fetchone()[0] == 1
        assert reader.execute("SELECT value FROM metadata WHERE key='reader-test'").fetchone()[0] == "ok"
        with pytest.raises(sqlite3.OperationalError):
            reader.execute("DELETE FROM metadata")
