import json
import sqlite3

import pytest
from axp_core.metadata import (IndexRebuildRequired, ensure_index_signature,
                               index_signature, validate_index_signature)
from axp_core.schema import EMBEDDING_SEMANTIC_VERSION


def database():
    con = sqlite3.connect(":memory:")
    con.executescript("CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT); CREATE TABLE chunks(id INTEGER); "
                      "CREATE TABLE schema_version(version INTEGER); INSERT INTO schema_version VALUES(4);")
    return con


def test_embedding_semantic_fingerprint_is_deterministic():
    first = index_signature("model", 384)
    assert first == index_signature("model", 384)
    assert first["embedding_semantic_version"] == EMBEDDING_SEMANTIC_VERSION


def test_known_incompatible_embedding_semantics_require_rebuild():
    con = database()
    incompatible = index_signature("model", 384)
    incompatible["embedding_semantic_version"] += 1
    con.execute("INSERT INTO metadata VALUES('index_signature',?)",
                (json.dumps(incompatible, sort_keys=True, separators=(",", ":")),))
    with pytest.raises(IndexRebuildRequired, match="Index rebuild required"):
        validate_index_signature(con, "model", 384)


def test_alpha5_signature_without_fingerprint_remains_provisionally_compatible():
    con = database()
    legacy = index_signature("model", 384)
    legacy.pop("embedding_semantic_version")
    encoded = json.dumps(legacy, sort_keys=True, separators=(",", ":"))
    con.execute("INSERT INTO metadata VALUES('index_signature',?)", (encoded,))
    validate_index_signature(con, "model", 384)
    assert ensure_index_signature(con, "model", 384) == legacy
    assert con.execute("SELECT value FROM metadata WHERE key='index_signature'").fetchone()[0] == encoded
