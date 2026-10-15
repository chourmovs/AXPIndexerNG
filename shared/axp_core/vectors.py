from .tracing import span, traced, sql_rows
import struct
import time
from contextlib import contextmanager


KNN_SQL = """SELECT rowid chunk_id,distance vector_distance FROM chunk_vectors
 WHERE embedding MATCH ? AND k=? ORDER BY distance"""

COMBINED_SQL = """SELECT v.rowid chunk_id,v.distance vector_distance,c.document_id,c.chunk_no,c.page_no,
 c.section_heading heading,d.path,d.filename,d.title,d.ingestion_mode,d.source_id,s.label source_label,s.path source_path,
 c.text snippet,c.identifiers
 FROM chunk_vectors v JOIN chunks c ON c.id=v.rowid JOIN documents d ON d.id=c.document_id
 JOIN sources s ON s.id=d.source_id
 WHERE embedding MATCH ? AND k=? ORDER BY distance"""


def _metadata_sql(size):
    return f"""SELECT c.id chunk_id,c.document_id,c.chunk_no,c.page_no,c.section_heading heading,
 d.path,d.filename,d.title,d.ingestion_mode,d.source_id,s.label source_label,s.path source_path,
 c.text snippet,c.identifiers FROM chunks c JOIN documents d ON d.id=c.document_id
 JOIN sources s ON s.id=d.source_id WHERE c.id IN ({','.join('?' for _ in range(size))})"""


@contextmanager
def read_snapshot(con):
    """Keep KNN and hydration on one snapshot, including inside a caller transaction."""
    con.execute("SAVEPOINT vector_read")
    try:
        yield
    except BaseException:
        con.execute("ROLLBACK TO vector_read")
        raise
    finally:
        con.execute("RELEASE vector_read")


def _fetch(con, sql, params, timings, stage):
    with span(stage, "sqlite.execute+fetchall"):
        start = time.perf_counter()
        with span("Exécution SQL", "sqlite.execute"):
            cursor = con.execute(sql, params)
        executed = time.perf_counter()
        with span("Lecture des lignes", "sqlite.Cursor.fetchall") as measured:
            rows = cursor.fetchall()
            measured.count(rows_read=len(rows))
        fetched = time.perf_counter()
    if timings is not None:
        timings[f"{stage}_execute_ms"] = (executed - start) * 1000
        timings[f"{stage}_fetchall_ms"] = (fetched - executed) * 1000
    return rows


def knn(con, vector, limit=20, *, timings=None):
    limit = min(max(0, int(limit)), 500)
    return ([dict(row) for row in _fetch(con, KNN_SQL, (serialize(vector), limit), timings, "vector_knn")]
            if limit else [])


def enrich(con, candidates, *, timings=None):
    """One bounded metadata query; retain native distances, ties and inner-join semantics."""
    if not candidates:
        return []
    ids = list(dict.fromkeys(row["chunk_id"] for row in candidates))
    if len(ids) > 500:
        raise ValueError("vector enrichment is bounded to 500 candidates")
    rows = _fetch(con, _metadata_sql(len(ids)), ids, timings, "vector_enrichment")
    by_id = {row["chunk_id"]: dict(row) for row in rows}
    return [{**by_id[row["chunk_id"]], "vector_distance": row["vector_distance"]}
            for row in candidates if row["chunk_id"] in by_id]


def search_split(con, vector, limit=20, *, timings=None):
    with read_snapshot(con):
        return enrich(con, knn(con, vector, limit, timings=timings), timings=timings)


@traced("Sérialisation du vecteur de requête")
def serialize(vector):
    return struct.pack(f"{len(vector)}f", *map(float, vector))


def upsert(con, chunk_id, vector):
    con.execute("DELETE FROM chunk_vectors WHERE rowid=?", (chunk_id,))
    con.execute("INSERT INTO chunk_vectors(rowid,embedding) VALUES (?,?)", (chunk_id, serialize(vector)))


@traced("Vecteurs : recherche native globale")
def search(con, vector, limit=20, *, timings=None):
    limit = min(max(0, int(limit)), 500)
    if not limit:
        return []
    rows = _fetch(con, COMBINED_SQL, (serialize(vector), limit), timings, "vector_combined")
    return [dict(r) for r in rows]


def developer_diagnostics(con, vector, limit=100):
    """Explicit opt-in only: schema, real SQL and EQP; never called by interactive search."""
    import re

    limit = min(max(1, int(limit)), 500)
    schema = [dict(row) for row in con.execute(
        "SELECT type,name,tbl_name,sql FROM sqlite_master WHERE name='chunk_vectors' "
        "OR (type='index' AND tbl_name IN ('chunks','documents','sources'))")]
    declaration = next((row["sql"] for row in schema if row["name"] == "chunk_vectors"), "")
    dimension = re.search(r"float\[(\d+)\]", declaration, re.I)
    dimension = int(dimension[1]) if dimension else None
    queries = {"knn": (KNN_SQL, (serialize(vector), limit)),
               "combined": (COMBINED_SQL, (serialize(vector), limit))}
    with read_snapshot(con):
        count = con.execute("SELECT count(*) FROM chunk_vectors").fetchone()[0]
        candidates = knn(con, vector, limit)
        if candidates:
            ids = [row["chunk_id"] for row in candidates]
            queries["enrichment"] = (_metadata_sql(len(ids)), ids)
        plans = {name: {"sql": sql, "plan": [dict(row) for row in con.execute(
            "EXPLAIN QUERY PLAN " + sql, params)]} for name, (sql, params) in queries.items()}
    metric = re.search(r"distance_metric\s*=\s*(\w+)", declaration, re.I)
    return {"sqlite_vec": con.execute("SELECT vec_version()").fetchone()[0],
            "schema": schema, "dimension": dimension, "metric": metric[1].lower() if metric else "l2 (vec0 default)",
            "vector_count": count, "useful_vector_bytes": count * dimension * 4 if dimension else None,
            "top_k": limit, "filters": ["embedding MATCH query", "k (native KNN bound)"],
            "joins": "inner chunks -> documents -> sources; no enabled/status/content filter",
            "vectors_visited": None, "queries": plans}


@traced('Vecteurs : distance native par document')
def search_documents(con, vector, document_ids):
    """Scan existing vectors for selected documents in one bounded SQL query."""
    ids = sorted({int(value) for value in document_ids})
    if not ids:
        return []
    placeholders = ",".join("?" for _ in ids)
    rows = sql_rows(con, f"""SELECT v.rowid chunk_id,coalesce(vec_distance_cosine(v.embedding,?),1.0) vector_distance,
 c.document_id,c.chunk_no,c.page_no,c.section_heading heading,d.path,d.filename,d.title,
 d.ingestion_mode,d.source_id,s.label source_label,s.path source_path,c.text snippet,c.identifiers
 FROM chunks c CROSS JOIN chunk_vectors v ON v.rowid=c.id
 JOIN documents d ON d.id=c.document_id JOIN sources s ON s.id=d.source_id
 WHERE c.document_id IN ({placeholders})""", (serialize(vector), *ids),
        name="Distances sqlite-vec et métadonnées par document")
    # Distance is evaluated by sqlite-vec in native code; no vector blobs cross
    # into Python and no repeated unpack/norm/dot conversion is needed.
    with span("Conversion et classement local", "vectors.local_sort") as measured:
        measured.count(vectors_visited=len(rows), candidates=len(rows))
        return sorted((dict(row) for row in rows), key=lambda row: (row["vector_distance"], row["chunk_id"]))
