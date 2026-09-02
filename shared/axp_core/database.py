import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path

from .metadata import IndexRebuildRequired, upgrade_v3_index_signature
from .schema import BASE_SCHEMA, PREVIOUS_SCHEMA_VERSION, SCHEMA_VERSION, SOURCES_SCHEMA
from .sources import default_label, detect_source_kind, normalize_source_path


class CapabilityError(RuntimeError):
    pass


def open_catalog_reader(path: str | Path, *, busy_timeout_ms: int = 1000):
    """Open a minimal, strictly read-only connection for dashboard queries."""
    db_path = Path(path).resolve()
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    con.execute(f"PRAGMA busy_timeout={max(0, int(busy_timeout_ms))}")
    con.execute("PRAGMA query_only=ON")
    return con


def connect(path: str | Path, *, dimension: int | None = None, readonly: bool = False,
            check_same_thread: bool = True):
    db_path = Path(path).resolve()
    target = f"file:{db_path}?mode=ro" if readonly else str(db_path)
    con = sqlite3.connect(target, uri=readonly, check_same_thread=check_same_thread)
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


SEARCH_CACHE_KIB = 65_536
SEARCH_MMAP_BYTES = 268_435_456


def configure_search_reader(con):
    """Apply bounded, read-intensive settings without affecting writer connections."""
    con.execute("PRAGMA query_only=ON")
    con.execute(f"PRAGMA cache_size=-{SEARCH_CACHE_KIB}")
    con.execute("PRAGMA temp_store=MEMORY")
    try:
        con.execute(f"PRAGMA mmap_size={SEARCH_MMAP_BYTES}")
    except sqlite3.Error:
        # mmap is an optional SQLite/platform facility; the page cache remains usable.
        pass
    return search_reader_diagnostics(con)


def fts_structural_diagnostics(con):
    """Return best-effort FTS5 segment/page counts across SQLite versions."""
    try:
        row = con.execute(
            "SELECT count(DISTINCT segid), count(DISTINCT segid || ':' || pgno) FROM chunks_fts_idx"
        ).fetchone()
        return {"fts_segment_count": int(row[0]), "fts_index_pages": int(row[1])}
    except sqlite3.Error:
        return {"fts_segment_count": None, "fts_index_pages": None}


def search_reader_diagnostics(con):
    """Collect safe settings and index-size diagnostics for field telemetry."""
    def pragma(name, default=None):
        try:
            row = con.execute(f"PRAGMA {name}").fetchone()
            return row[0] if row else default
        except sqlite3.Error:
            return default

    def count(table):
        try:
            return int(con.execute(f"SELECT count(*) FROM {table}").fetchone()[0])
        except sqlite3.Error:
            return None

    return {
        "sqlite_cache_size": pragma("cache_size"),
        "sqlite_mmap_size": pragma("mmap_size", 0),
        "sqlite_temp_store": pragma("temp_store"),
        "total_chunks": count("chunks"),
        "total_vectors": count("chunk_vectors"),
        **fts_structural_diagnostics(con),
    }


class SearchReaderPool:
    """Lazy bounded pool whose SQLite readers are exclusively borrowed."""

    def __init__(self, path: str | Path, size: int = 2):
        if int(size) < 1:
            raise ValueError("Search reader pool size must be positive")
        self.path = Path(path)
        self.size = int(size)
        self._available = []
        self._created = 0
        self._closed = False
        self._condition = threading.Condition()

    @contextmanager
    def acquire(self):
        con = None
        reused = False
        with self._condition:
            while not self._closed and not self._available and self._created >= self.size:
                self._condition.wait()
            if self._closed:
                raise RuntimeError("Search reader pool is closed")
            if self._available:
                con = self._available.pop()
                reused = True
            else:
                # Reserve the slot before opening so concurrent initialization stays bounded.
                self._created += 1
        if con is None:
            try:
                con = connect(self.path, readonly=True, check_same_thread=False)
                configure_search_reader(con)
            except Exception:
                with self._condition:
                    self._created -= 1
                    self._condition.notify()
                raise
        try:
            yield con, reused
        finally:
            # End any accidental snapshot before another request borrows this reader.
            if con.in_transaction:
                con.rollback()
            with self._condition:
                if self._closed:
                    con.close()
                    self._created -= 1
                else:
                    self._available.append(con)
                self._condition.notify()

    def close(self):
        """Reject new borrowers and close idle readers (borrowed readers close on return)."""
        with self._condition:
            self._closed = True
            readers, self._available = self._available, []
            self._created -= len(readers)
            self._condition.notify_all()
        for con in readers:
            con.close()


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
        signature_row = con.execute("SELECT value FROM metadata WHERE key='index_signature'").fetchone()
        if signature_row:
            upgraded_signature = upgrade_v3_index_signature(signature_row[0])
            if upgraded_signature is not None:
                con.execute("UPDATE metadata SET value=? WHERE key='index_signature'", (upgraded_signature,))
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
