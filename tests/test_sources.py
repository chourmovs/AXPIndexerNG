import json
import sqlite3

import pytest
import sqlite_vec
from axp_core.database import _migrate_v2_to_v3, connect
from axp_core.metadata import IndexRebuildRequired, ensure_index_signature, index_signature
from axp_core.sources import (
    SourceOverlapError,
    add_source,
    disable_source,
    enable_source,
    list_sources,
    normalize_source_path,
    remove_source,
)
from axp_core.vectors import upsert
from axp_tray.sources_window import add_gui_source


def test_windows_path_normalization_and_source_kinds(tmp_path):
    con = connect(tmp_path / "sources.db", dimension=3)
    drive = add_source(con, "D:/")
    unc = add_source(con, r"\\SERVER\Documentation")
    folder = add_source(con, r"E:\Projects")
    assert (drive["path"], drive["path_key"], drive["kind"]) == ("D:\\", "d:\\", "drive")
    assert unc["kind"] == "unc" and unc["path_key"] == r"\\server\documentation"
    assert folder["kind"] == "folder"
    assert normalize_source_path(r"C:\Users\User\Documents\\")[0] == r"C:\Users\User\Documents"


def test_gui_selected_source_is_explicitly_recursive(tmp_path):
    con = connect(tmp_path / "gui-source.db", dimension=3)

    source = add_gui_source(con, tmp_path / "selected")

    assert source["recursive"] == 1
    assert con.execute("SELECT recursive FROM sources WHERE id=?", (source["id"],)).fetchone()[0] == 1


def test_repository_defaults_recursive_but_preserves_explicit_false(tmp_path):
    con = connect(tmp_path / "recursion-policy.db", dimension=3)

    default_source = add_source(con, tmp_path / "default")
    flat_source = add_source(con, tmp_path / "flat", recursive=False)

    assert default_source["recursive"] == 1
    assert flat_source["recursive"] == 0


def test_duplicates_overlap_and_false_prefix(tmp_path):
    con = connect(tmp_path / "overlap.db", dimension=3)
    add_source(con, r"D:\Process")
    with pytest.raises(SourceOverlapError, match="already covered"):
        add_source(con, r"d:\PROCESS")
    with pytest.raises(SourceOverlapError, match="already covered"):
        add_source(con, r"D:\Process\Batch")
    with pytest.raises(SourceOverlapError, match="would subsume"):
        add_source(con, "D:\\")
    # Prefixes which are not complete path components do not overlap.
    add_source(con, r"D:\ProcessArchive")
    assert len(list_sources(con)) == 2


def test_enable_disable_and_scoped_cascade(tmp_path):
    con = connect(tmp_path / "cascade.db", dimension=3)
    a = add_source(con, tmp_path / "a")
    b = add_source(con, tmp_path / "b")
    chunk_ids = {}
    for source, key in ((a, "a"), (b, "b")):
        doc = con.execute(
            "INSERT INTO documents(source_id,path,path_key,extension,size_bytes,modified_unix_ms,sha256,indexed_unix_ms) "
            "VALUES(?,?,?,?,1,1,'x',1)", (source["id"], key, key, ".txt")
        ).lastrowid
        chunk_ids[key] = con.execute("INSERT INTO chunks(document_id,chunk_no,text) VALUES(?,0,?)", (doc, key)).lastrowid
        upsert(con, chunk_ids[key], [1, 0, 0])
    con.commit()
    assert disable_source(con, a["id"])["status"] == "disabled"
    assert enable_source(con, a["id"])["status"] == "idle"
    remove_source(con, a["id"])
    assert con.execute("SELECT count(*) FROM documents WHERE source_id=?", (a["id"],)).fetchone()[0] == 0
    assert con.execute("SELECT count(*) FROM documents WHERE source_id=?", (b["id"],)).fetchone()[0] == 1
    assert not con.execute("SELECT 1 FROM chunk_vectors WHERE rowid=?", (chunk_ids["a"],)).fetchone()
    assert con.execute("SELECT 1 FROM chunk_vectors WHERE rowid=?", (chunk_ids["b"],)).fetchone()


def test_schema_v3_forward_migration_preserves_index_and_upgrades_signature(tmp_path):
    db = tmp_path / "alpha3.db"
    con = sqlite3.connect(db)
    con.enable_load_extension(True)
    sqlite_vec.load(con)
    con.executescript("""
        CREATE TABLE schema_version(version INTEGER NOT NULL);
        INSERT INTO schema_version VALUES(2);
        CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);
        INSERT INTO metadata VALUES('index_signature','placeholder');
        CREATE TABLE documents(id INTEGER PRIMARY KEY,source_root TEXT NOT NULL,path TEXT NOT NULL,path_key TEXT NOT NULL UNIQUE,
          extension TEXT NOT NULL,size_bytes INTEGER NOT NULL,modified_unix_ms INTEGER NOT NULL,sha256 TEXT NOT NULL,
          indexed_unix_ms INTEGER NOT NULL,title TEXT NOT NULL DEFAULT '',filename TEXT NOT NULL DEFAULT '');
        CREATE TABLE chunks(id INTEGER PRIMARY KEY,document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
          chunk_no INTEGER NOT NULL,text TEXT NOT NULL,page_no INTEGER,char_start INTEGER,char_end INTEGER,
          section_heading TEXT NOT NULL DEFAULT '',identifiers TEXT NOT NULL DEFAULT '',UNIQUE(document_id,chunk_no));
        CREATE VIRTUAL TABLE chunks_fts USING fts5(text,title,filename,heading,identifiers,content='',contentless_delete=1);
        CREATE VIRTUAL TABLE chunk_vectors USING vec0(embedding float[3] distance_metric=cosine);
        INSERT INTO documents VALUES(1,'D:\\Process','D:\\Process\\reactor.txt','d:\\process\\reactor.txt','.txt',1,1,'hash',1,'Reactor','reactor.txt');
        INSERT INTO chunks VALUES(1,1,0,'reactor pressure',NULL,0,16,'','');
        INSERT INTO chunks_fts(rowid,text,title,filename,heading,identifiers) VALUES(1,'reactor pressure','Reactor','reactor.txt','','');
        INSERT INTO chunk_vectors(rowid,embedding) VALUES(1,vec_f32('[1,0,0]'));
    """)
    signature = index_signature("test-model", 3)
    signature["schema_version"] = 3
    con.execute(
        "UPDATE metadata SET value=? WHERE key='index_signature'",
        (json.dumps(signature, sort_keys=True, separators=(",", ":")),),
    )
    con.commit()
    _migrate_v2_to_v3(con)
    con.close()
    migrated = connect(db, dimension=3)
    assert migrated.execute("SELECT version FROM schema_version").fetchone()[0] == 4
    assert migrated.execute("SELECT count(*) FROM documents").fetchone()[0] == 1
    assert migrated.execute("SELECT count(*) FROM chunks").fetchone()[0] == 1
    assert migrated.execute("SELECT count(*) FROM documents WHERE source_id IS NOT NULL").fetchone()[0] == 1
    assert migrated.execute("SELECT count(*) FROM chunks_fts").fetchone()[0] == 1
    assert migrated.execute("SELECT count(*) FROM chunk_vectors").fetchone()[0] == 1
    stored_signature = json.loads(
        migrated.execute("SELECT value FROM metadata WHERE key='index_signature'").fetchone()[0]
    )
    assert stored_signature["schema_version"] == 4
    ensure_index_signature(migrated, "test-model", 3)
    assert migrated.execute("SELECT recursive FROM sources").fetchone()[0] == 1
    assert migrated.execute("SELECT ingestion_mode FROM documents").fetchone()[0] == "content"


def _populated_database_with_v3_signature(tmp_path):
    con = connect(tmp_path / "already-migrated.db", dimension=3)
    source = add_source(con, tmp_path / "indexed")
    document_id = con.execute(
        "INSERT INTO documents(source_id,path,path_key,extension,size_bytes,modified_unix_ms,sha256,indexed_unix_ms) "
        "VALUES(?,?,?,?,1,1,'hash',1)",
        (source["id"], "item.txt", "item.txt", ".txt"),
    ).lastrowid
    chunk_id = con.execute(
        "INSERT INTO chunks(document_id,chunk_no,text) VALUES(?,0,'preserved text')", (document_id,)
    ).lastrowid
    upsert(con, chunk_id, [1, 0, 0])
    signature = index_signature("test-model", 3)
    signature["schema_version"] = 3
    con.execute(
        "INSERT INTO metadata(key,value) VALUES('index_signature',?)",
        (json.dumps(signature, sort_keys=True, separators=(",", ":")),),
    )
    con.commit()
    return con, chunk_id


def test_already_migrated_database_repairs_compatible_signature(tmp_path, caplog):
    con, chunk_id = _populated_database_with_v3_signature(tmp_path)

    with caplog.at_level("INFO", logger="axp_core"):
        ensure_index_signature(con, "test-model", 3)

    assert json.loads(con.execute("SELECT value FROM metadata WHERE key='index_signature'").fetchone()[0])[
        "schema_version"
    ] == 4
    assert con.execute("SELECT text FROM chunks WHERE id=?", (chunk_id,)).fetchone()[0] == "preserved text"
    assert con.execute("SELECT count(*) FROM chunk_vectors WHERE rowid=?", (chunk_id,)).fetchone()[0] == 1
    assert "Upgraded compatible index signature schema 3 -> 4" in caplog.text


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("embedding_model_id", "other-model"),
        ("embedding_dimension", 99),
        ("distance_metric", "euclidean"),
        ("chunker_version", 99),
        ("embedding_input_version", 99),
        ("unknown", "field"),
        ("schema_version", 2),
    ],
)
def test_v3_signature_repair_rejects_every_incompatibility(tmp_path, field, value):
    con, _ = _populated_database_with_v3_signature(tmp_path)
    signature = json.loads(con.execute("SELECT value FROM metadata WHERE key='index_signature'").fetchone()[0])
    signature[field] = value
    con.execute(
        "UPDATE metadata SET value=? WHERE key='index_signature'",
        (json.dumps(signature, sort_keys=True, separators=(",", ":")),),
    )
    con.commit()

    with pytest.raises(IndexRebuildRequired, match="runtime index signature does not match"):
        ensure_index_signature(con, "test-model", 3)
