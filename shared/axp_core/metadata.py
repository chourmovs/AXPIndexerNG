import json

from .schema import CHUNKER_VERSION, DISTANCE_METRIC, EMBEDDING_INPUT_VERSION, SCHEMA_VERSION


class IndexRebuildRequired(RuntimeError):
    pass


def read_index_signature(con):
    """Return and minimally validate the dense configuration recorded by the index."""
    row = con.execute("SELECT value FROM metadata WHERE key='index_signature'").fetchone()
    if not row:
        raise IndexRebuildRequired("Index rebuild required: database has no index signature")
    try:
        signature = json.loads(row[0])
        model_id = signature["embedding_model_id"]
        dimension = int(signature["embedding_dimension"])
        metric = signature["distance_metric"]
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise IndexRebuildRequired("Index rebuild required: database has an invalid index signature") from exc
    if not isinstance(model_id, str) or not model_id or dimension <= 0 or metric != DISTANCE_METRIC:
        raise IndexRebuildRequired("Index rebuild required: database has an incompatible index signature")
    return signature


def index_signature(model_id, dimension, distance_metric=DISTANCE_METRIC):
    return {
        "schema_version": SCHEMA_VERSION,
        "embedding_model_id": model_id,
        "embedding_dimension": int(dimension),
        "distance_metric": distance_metric,
        "chunker_version": CHUNKER_VERSION,
        "embedding_input_version": EMBEDDING_INPUT_VERSION,
    }


def ensure_index_signature(con, model_id, dimension, distance_metric=DISTANCE_METRIC):
    wanted = index_signature(model_id, dimension, distance_metric)
    encoded = json.dumps(wanted, sort_keys=True, separators=(",", ":"))
    row = con.execute("SELECT value FROM metadata WHERE key='index_signature'").fetchone()
    chunks = con.execute("SELECT count(*) FROM chunks").fetchone()[0]
    if row and row[0] != encoded:
        raise IndexRebuildRequired("Index rebuild required: runtime index signature does not match the database")
    if not row and chunks:
        raise IndexRebuildRequired("Index rebuild required: database has no compatible index signature")
    con.execute("INSERT OR REPLACE INTO metadata(key,value) VALUES('index_signature',?)", (encoded,))
    con.commit()
    return wanted


def validate_index_signature(con, model_id, dimension, distance_metric=DISTANCE_METRIC):
    wanted = json.dumps(index_signature(model_id, dimension, distance_metric), sort_keys=True, separators=(",", ":"))
    row = con.execute("SELECT value FROM metadata WHERE key='index_signature'").fetchone()
    if not row or row[0] != wanted:
        raise IndexRebuildRequired("Index rebuild required: runtime index signature does not match the database")


def ensure_model(con, model_id, dimension, revision="unknown"):
    return ensure_index_signature(con, model_id, dimension)
