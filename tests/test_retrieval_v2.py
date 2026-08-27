import json

import pytest

from axp_client.reranker import EmbeddingLRU, maxsim
from axp_core.fts import build_query
from axp_core.hybrid import diversify
from axp_core.identifiers import extract_identifiers
from axp_core.metadata import IndexRebuildRequired, ensure_index_signature
from axp_daemon.chunker import chunk_text


def test_chunking_is_structural_deterministic_and_propagates_heading():
    text = "PRESSURE CONTROL\n\n" + "A complete pressure sentence. " * 80 + "\n\nA final paragraph."
    chunks = chunk_text(text, page_no=17, target_words=40, overlap_words=10, max_words=60)
    assert chunks == chunk_text(text, page_no=17, target_words=40, overlap_words=10, max_words=60)
    assert all(chunk.page_no == 17 and chunk.section_heading == "PRESSURE CONTROL" for chunk in chunks)
    assert all(chunk.text.endswith((".", "CONTROL")) for chunk in chunks)


def test_identifier_normalization_and_safe_query():
    identifiers = dict(extract_identifiers("R421, R-042500, F040100 and SOP-1234"))
    assert set(identifiers) == {"R421", "R042500", "F040100", "SOP1234"}
    built = build_query('\"R-042500\" (pression/réacteur): wild*')
    assert '"R-042500"' in built and '"R042500"' in built
    assert "(" not in built and "*" not in built and ":" not in built


def test_signature_mismatch_fails_loudly():
    class Connection:
        value = None

        def execute(self, sql, parameters=()):
            if "SELECT value" in sql:
                return self
            if "count" in sql:
                self.value = (0,)
                return self
            return self

        def fetchone(self):
            value, self.value = self.value, None
            return value

        def commit(self):
            pass

    con = Connection()
    ensure_index_signature(con, "balanced", 384)
    con.value = (json.dumps({"wrong": True}),)
    with pytest.raises(IndexRebuildRequired, match="rebuild required"):
        ensure_index_signature(con, "quality", 1024)


def test_maxsim_cache_and_diversification():
    assert maxsim([[1, 0], [0, 1]], [[1, 0], [0, 0.5]]) == 1.5
    cache = EmbeddingLRU(2)
    for key in ("a", "b", "c"):
        cache.put(key, key)
    assert cache.get("a") is None and cache.get("c") == "c"
    rows = [{"document_id": 1, "n": n} for n in range(4)] + [{"document_id": 2, "n": 9}]
    assert [row["document_id"] for row in diversify(rows, 4, 3)] == [1, 2, 1, 1]
