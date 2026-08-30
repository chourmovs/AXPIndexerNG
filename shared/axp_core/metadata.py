import json
import logging

from .schema import (CHUNKER_VERSION, DISTANCE_METRIC, EMBEDDING_INPUT_VERSION,
                     EMBEDDING_SEMANTIC_VERSION, SCHEMA_VERSION)


class IndexRebuildRequired(RuntimeError):
    pass


LOGGER = logging.getLogger("axp_core")
_SIGNATURE_FIELDS = {
    "schema_version",
    "embedding_model_id",
    "embedding_dimension",
    "distance_metric",
    "chunker_version",
    "embedding_input_version",
    "embedding_semantic_version",
}
_SEMANTIC_FIELDS = _SIGNATURE_FIELDS - {"schema_version"}


def _canonical_signature(signature):
    return json.dumps(signature, sort_keys=True, separators=(",", ":"))


def _is_well_formed_signature(signature):
    fields = set(signature) if isinstance(signature, dict) else set()
    return (
        isinstance(signature, dict)
        and fields in (_SIGNATURE_FIELDS, _SIGNATURE_FIELDS - {"embedding_semantic_version"})
        and type(signature["schema_version"]) is int
        and isinstance(signature["embedding_model_id"], str)
        and bool(signature["embedding_model_id"])
        and type(signature["embedding_dimension"]) is int
        and signature["embedding_dimension"] > 0
        and isinstance(signature["distance_metric"], str)
        and type(signature["chunker_version"]) is int
        and type(signature["embedding_input_version"]) is int
        and ("embedding_semantic_version" not in signature
             or type(signature["embedding_semantic_version"]) is int)
    )


def upgrade_v3_index_signature(encoded):
    """Return a canonical v4 encoding for the one known lossless signature transition."""
    try:
        stored = json.loads(encoded)
    except (TypeError, json.JSONDecodeError):
        return None
    if not _is_well_formed_signature(stored) or stored["schema_version"] != 3:
        return None
    if (
        stored["distance_metric"] != DISTANCE_METRIC
        or stored["chunker_version"] != CHUNKER_VERSION
        or stored["embedding_input_version"] != EMBEDDING_INPUT_VERSION
    ):
        return None
    return _canonical_signature({**stored, "schema_version": SCHEMA_VERSION})


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
        "embedding_semantic_version": EMBEDDING_SEMANTIC_VERSION,
    }


def _legacy_alpha5_signature(signature, wanted):
    """Recognize the pre-fingerprint signature without claiming its pooling history."""
    return (isinstance(signature, dict)
            and set(signature) == _SIGNATURE_FIELDS - {"embedding_semantic_version"}
            and all(signature.get(field) == wanted[field]
                    for field in _SEMANTIC_FIELDS - {"embedding_semantic_version"}))


def ensure_index_signature(con, model_id, dimension, distance_metric=DISTANCE_METRIC):
    wanted = index_signature(model_id, dimension, distance_metric)
    encoded = _canonical_signature(wanted)
    row = con.execute("SELECT value FROM metadata WHERE key='index_signature'").fetchone()
    chunks = con.execute("SELECT count(*) FROM chunks").fetchone()[0]
    if row and row[0] != encoded:
        try:
            stored = json.loads(row[0])
            schema_row = con.execute("SELECT version FROM schema_version").fetchone()
        except (TypeError, json.JSONDecodeError):
            stored = None
            schema_row = None
        if _legacy_alpha5_signature(stored, wanted):
            # Preserve existing alpha5 vectors while pooling history is investigated.
            # Do not stamp an unproven semantic revision onto the database.
            return stored
        compatible_upgrade = (
            schema_row is not None
            and schema_row[0] == SCHEMA_VERSION
            and _is_well_formed_signature(stored)
            and stored["schema_version"] == 3
            and all(stored[field] == wanted[field]
                    for field in _SEMANTIC_FIELDS if field in stored)
        )
        if not compatible_upgrade:
            raise IndexRebuildRequired("Index rebuild required: runtime index signature does not match the database")
        con.execute("UPDATE metadata SET value=? WHERE key='index_signature'", (encoded,))
        con.commit()
        LOGGER.info("Upgraded compatible index signature schema 3 -> 4; existing vectors preserved")
        return wanted
    if not row and chunks:
        raise IndexRebuildRequired("Index rebuild required: database has no compatible index signature")
    con.execute("INSERT OR REPLACE INTO metadata(key,value) VALUES('index_signature',?)", (encoded,))
    con.commit()
    return wanted


def validate_index_signature(con, model_id, dimension, distance_metric=DISTANCE_METRIC):
    wanted_signature = index_signature(model_id, dimension, distance_metric)
    wanted = _canonical_signature(wanted_signature)
    row = con.execute("SELECT value FROM metadata WHERE key='index_signature'").fetchone()
    try:
        stored = json.loads(row[0]) if row else None
    except (TypeError, json.JSONDecodeError):
        stored = None
    if not row or (row[0] != wanted and not _legacy_alpha5_signature(stored, wanted_signature)):
        raise IndexRebuildRequired("Index rebuild required: runtime index signature does not match the database")


def ensure_model(con, model_id, dimension, revision="unknown"):
    return ensure_index_signature(con, model_id, dimension)
