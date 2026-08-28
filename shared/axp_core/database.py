import sqlite3
import time
from pathlib import Path

from .metadata import IndexRebuildRequired
from .schema import BASE_SCHEMA, PREVIOUS_SCHEMA_VERSION, SCHEMA_VERSION, SOURCES_SCHEMA
from .sources import default_label, detect_source_kind, normalize_source_path


class CapabilityError(RuntimeError):
    pass


def connect(path: str | Path, *, dimension: int | None = None, readonly: bool = False):
    db_path = Path(path).resolve()
    target = f"file:{db_path}?mode=ro" if readonly else str(db_path)
    con = sqlite3.connect(target, uri=readonly)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys=ON")
    con.execute("PRAGMA busy_timeout=5000")
    if not readonly:
        con.execute("PRAGMA journal_mode=WAL")
    try:
        con.execute("CREATE VIRTUAL TABLE temp.fts_probe USING fts5(value)")
    except sqlite3.Error as exc:
        raise CapabilityError("SQLite FTS5 is required") from exc
    load_vectors(con)
    if readonly:
        _check_version(con)
        return con
    # Supported schemas migrate losslessly and sequentially.
    if con.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='schema_version'").fetchone():
        row = con.execute("SELECT version FROM schema_version").fetchone()
        if row and row[0] == 2:
            _migrate_v2_to_v3(con)
            row = con.execute("SELECT version FROM schema_version").fetchone()
        if row and row[0] == PREVIOUS_SCHEMA_VERSION:
            _migrate_v3_to_v4(con)
        else:
            _check_version(con)
    if dimension is not None:
        con.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS chunk_vectors USING vec0(embedding float[{int(dimension)}] distance_metric=cosine)"
        )
    con.executescript(BASE_SCHEMA)
    if dimension is not None:
        con.execute("CREATE TRIGGER IF NOT EXISTS chunks_vector_delete AFTER DELETE ON chunks BEGIN DELETE FROM chunk_vectors WHERE rowid=old.id; END")
    row = con.execute("SELECT version FROM schema_version").fetchone()
    if row is None:
        con.execute("INSERT INTO schema_version VALUES (?)", (SCHEMA_VERSION,))
    con.commit()
    return con


def _check_version(con):
    try:
        row = con.execute("SELECT version FROM schema_version").fetchone()
    except sqlite3.Error as exc:
        raise IndexRebuildRequired("Index rebuild required: unrecognized database schema") from exc
    if row is None or row[0] != SCHEMA_VERSION:
        value = "missing" if row is None else row[0]
        raise IndexRebuildRequired(f"Index rebuild required: schema version {value} is incompatible")


def _migrate_v2_to_v3(con):
    """Add administrative sources while preserving documents, FTS and vectors."""
    now = int(time.time() * 1000)
    try:
        con.execute("BEGIN IMMEDIATE")
        con.execute(SOURCES_SCHEMA)
        columns = {row[1] for row in con.execute("PRAGMA table_info(documents)")}
        if "source_id" not in columns:
            con.execute("ALTER TABLE documents ADD COLUMN source_id INTEGER REFERENCES sources(id) ON DELETE CASCADE")
        roots = [row[0] for row in con.execute("SELECT DISTINCT source_root FROM documents ORDER BY source_root")]
        for root in roots:
            display, key = normalize_source_path(root)
            con.execute(
                """INSERT OR IGNORE INTO sources(
                path,path_key,label,kind,enabled,recursive,status,created_ms,updated_ms)
                VALUES(?,?,?,?,1,1,'idle',?,?)""",
                (display, key, default_label(display), detect_source_kind(display), now, now),
            )
            source_id = con.execute("SELECT id FROM sources WHERE path_key=?", (key,)).fetchone()[0]
            con.execute("UPDATE documents SET source_id=? WHERE source_root=?", (source_id, root))
        con.execute("CREATE INDEX IF NOT EXISTS documents_source ON documents(source_id)")
        con.execute("UPDATE schema_version SET version=3")
        con.commit()
    except Exception:
        con.rollback()
        raise


def _migrate_v3_to_v4(con):
    """Add ingestion and coverage metadata without rebuilding FTS or vector tables."""
    source_columns = {
        "last_seen_count": "INTEGER NOT NULL DEFAULT 0",
        "last_content_count": "INTEGER NOT NULL DEFAULT 0",
        "last_metadata_count": "INTEGER NOT NULL DEFAULT 0",
        "last_ignored_count": "INTEGER NOT NULL DEFAULT 0",
        "last_failed_count": "INTEGER NOT NULL DEFAULT 0",
        "last_extension_breakdown": "TEXT NOT NULL DEFAULT '{}'",
    }
    try:
        con.execute("BEGIN IMMEDIATE")
        existing = {row[1] for row in con.execute("PRAGMA table_info(sources)")}
        for name, declaration in source_columns.items():
            if name not in existing:
                con.execute(f"ALTER TABLE sources ADD COLUMN {name} {declaration}")
        document_columns = {row[1] for row in con.execute("PRAGMA table_info(documents)")}
        if "ingestion_mode" not in document_columns:
            con.execute(
                "ALTER TABLE documents ADD COLUMN ingestion_mode TEXT NOT NULL DEFAULT 'content' "
                "CHECK(ingestion_mode IN ('content','metadata'))"
            )
        con.execute("UPDATE schema_version SET version=?", (SCHEMA_VERSION,))
        con.commit()
    except Exception:
        con.rollback()
        raise


def rebuild(path, dimension):
    db = Path(path)
    for suffix in ("", "-wal", "-shm"):
        candidate = Path(str(db) + suffix)
        if candidate.exists():
            candidate.unlink()
    return connect(path, dimension=dimension)


def load_vectors(con):
    try:
        import sqlite_vec

        con.enable_load_extension(True)
        sqlite_vec.load(con)
        con.enable_load_extension(False)
        return con.execute("SELECT vec_version()").fetchone()[0]
    except Exception as exc:
        raise CapabilityError("sqlite-vec is required and could not be loaded") from exc


def capability_report(con):
    return {
        "sqlite": sqlite3.sqlite_version,
        "fts5": bool(con.execute("SELECT sqlite_compileoption_used('ENABLE_FTS5')").fetchone()[0]),
        "sqlite_vec": con.execute("SELECT vec_version()").fetchone()[0],
    }
