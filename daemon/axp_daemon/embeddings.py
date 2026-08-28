import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    dimension: int
    query_prefix: str = ""
    document_prefix: str = ""
    distance_metric: str = "cosine"


BALANCED = ModelSpec("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", 384)
QUALITY = ModelSpec("intfloat/multilingual-e5-large", 1024, "query: ", "passage: ")
PROFILES = {"balanced": BALANCED, "quality": QUALITY}
FASTEMBED_MODEL = os.getenv("FASTEMBED_MODEL", BALANCED.model_id)


def model_spec(value):
    if isinstance(value, ModelSpec):
        return value
    if value in PROFILES:
        return PROFILES[value]
    for spec in PROFILES.values():
        if value == spec.model_id:
            return spec
    return ModelSpec(value, 0)


class Embedder:
    def __init__(self, model_id=FASTEMBED_MODEL, cache_dir=None, local_only=True, runtime_batch_size=16):
        from fastembed import TextEmbedding

        self.spec = model_spec(model_id)
        self.model_id = self.spec.model_id
        self.runtime_batch_size = max(1, int(runtime_batch_size))
        kwargs = {"model_name": self.model_id}
        if cache_dir:
            kwargs["cache_dir"] = cache_dir
        if local_only:
            kwargs["local_files_only"] = True
        self._model = TextEmbedding(**kwargs)
        measured = len(next(iter(self._embed([self.spec.document_prefix + "dimension probe"]))))
        if self.spec.dimension and measured != self.spec.dimension:
            raise RuntimeError(f"Embedding model dimension mismatch: expected {self.spec.dimension}, got {measured}")
        self._dimension = measured

    def _embed(self, texts):
        return self._model.embed(texts, batch_size=self.runtime_batch_size)

    @property
    def dimension(self):
        return self._dimension

    @property
    def distance_metric(self):
        return self.spec.distance_metric

    def embed_documents(self, texts):
        return [list(x) for x in self._embed([self.spec.document_prefix + x for x in texts])]

    def embed_query(self, text):
        return list(next(iter(self._embed([self.spec.query_prefix + text]))))


def embedder_for_index(con, cache_dir=None, local_only=True):
    """Construct the exact dense query model declared by a database index."""
    from axp_core.metadata import IndexRebuildRequired, read_index_signature

    signature = read_index_signature(con)
    model_id = signature["embedding_model_id"]
    expected_dimension = int(signature["embedding_dimension"])
    spec = model_spec(model_id)
    if not spec.dimension:
        raise IndexRebuildRequired(
            f"Index uses unsupported embedding model {model_id!r}; configure a compatible client model"
        )
    if spec.dimension != expected_dimension:
        raise IndexRebuildRequired(
            f"Index embedding configuration is inconsistent: {model_id!r} requires {spec.dimension} dimensions, "
            f"but the database records {expected_dimension}"
        )
    try:
        embedder = Embedder(spec, cache_dir=cache_dir, local_only=local_only)
    except Exception as exc:
        raise RuntimeError(
            f"Required index embedding model {model_id!r} ({expected_dimension} dimensions) is not available "
            "in the local model cache"
        ) from exc
    if embedder.dimension != expected_dimension:
        raise IndexRebuildRequired(
            f"Index requires a {expected_dimension}-dimensional query model, got {embedder.dimension}"
        )
    return embedder
