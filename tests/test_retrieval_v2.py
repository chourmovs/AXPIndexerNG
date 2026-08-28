import json

import pytest
from axp_client.reranker import EmbeddingLRU, maxsim
from axp_core.database import connect
from axp_core.fts import build_query
from axp_core.hybrid import diversify, search
from axp_core.identifiers import extract_identifiers
from axp_core.metadata import IndexRebuildRequired, ensure_index_signature
from axp_core.vectors import upsert
from axp_daemon import embeddings
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


def test_quality_result_normalizes_numpy_scores_for_json(tmp_path):
    np = pytest.importorskip("numpy")
    con = connect(tmp_path / "quality.db", dimension=3)
    from axp_core.sources import add_source

    source_id = add_source(con, tmp_path / "root")["id"]
    document_id = con.execute(
        "INSERT INTO documents(source_id,path,path_key,extension,size_bytes,modified_unix_ms,sha256,"
        "indexed_unix_ms,title,filename) VALUES(?,'reactor.txt','reactor','.txt',1,1,'x',1,'Reactor','reactor.txt')",
        (source_id,),
    ).lastrowid
    chunk_id = con.execute(
        "INSERT INTO chunks(document_id,chunk_no,text) VALUES(?,0,'reactor pressure control')", (document_id,)
    ).lastrowid
    con.execute(
        "INSERT INTO chunks_fts(rowid,text,title,filename,heading,identifiers) VALUES(?,?,?,?,?,?)",
        (chunk_id, "reactor pressure control", "Reactor", "reactor.txt", "", ""),
    )
    upsert(con, chunk_id, [1.0, 0.0, 0.0])
    con.commit()

    class Float32Reranker:
        def score(self, query, candidates):
            return [np.float32(0.75) for _ in candidates]

    result = search(
        con, "reactor", [1.0, 0.0, 0.0], profile="quality", reranker=Float32Reranker(), explain=True
    )
    assert type(result["results"][0]["reranker_score"]) is float
    json.dumps(result)


@pytest.mark.parametrize(
    ("profile", "model_id", "dimension"),
    [("balanced", embeddings.BALANCED.model_id, 384), ("quality", embeddings.QUALITY.model_id, 1024)],
)
def test_client_uses_index_dense_configuration(tmp_path, monkeypatch, profile, model_id, dimension):
    con = connect(tmp_path / f"{profile}.db")
    ensure_index_signature(con, model_id, dimension)

    class StubEmbedder:
        def __init__(self, spec, **kwargs):
            self.model_id = spec.model_id
            self.dimension = spec.dimension
            self.distance_metric = spec.distance_metric
            self.spec = spec

    monkeypatch.setattr(embeddings, "Embedder", StubEmbedder)
    selected = embeddings.embedder_for_index(con)
    assert selected.model_id == model_id
    assert selected.dimension == dimension
    assert selected.spec.query_prefix == embeddings.PROFILES[profile].query_prefix


def test_client_rejects_incompatible_or_missing_dense_model(tmp_path, monkeypatch):
    con = connect(tmp_path / "incompatible.db")
    ensure_index_signature(con, embeddings.QUALITY.model_id, 384)
    with pytest.raises(IndexRebuildRequired, match="requires 1024 dimensions"):
        embeddings.embedder_for_index(con)

    con = connect(tmp_path / "missing.db")
    ensure_index_signature(con, embeddings.BALANCED.model_id, 384)

    class MissingEmbedder:
        def __init__(self, *args, **kwargs):
            raise FileNotFoundError("not cached")

    monkeypatch.setattr(embeddings, "Embedder", MissingEmbedder)
    with pytest.raises(RuntimeError, match="not available in the local model cache"):
        embeddings.embedder_for_index(con)
