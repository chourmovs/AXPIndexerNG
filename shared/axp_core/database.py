import sqlite3
from pathlib import Path

from .metadata import IndexRebuildRequired
from .schema import BASE_SCHEMA, SCHEMA_VERSION


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
    # Refuse old populated schemas rather than allowing CREATE IF NOT EXISTS to disguise them.
    if con.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='schema_version'").fetchone():
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
