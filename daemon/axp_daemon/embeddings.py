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
    def __init__(self, model_id=FASTEMBED_MODEL, cache_dir=None, local_only=True):
        from fastembed import TextEmbedding

        self.spec = model_spec(model_id)
        self.model_id = self.spec.model_id
        kwargs = {"model_name": self.model_id}
        if cache_dir:
            kwargs["cache_dir"] = cache_dir
        if local_only:
            kwargs["local_files_only"] = True
        self._model = TextEmbedding(**kwargs)
        measured = len(next(iter(self._model.embed([self.spec.document_prefix + "dimension probe"]))))
        if self.spec.dimension and measured != self.spec.dimension:
            raise RuntimeError(f"Embedding model dimension mismatch: expected {self.spec.dimension}, got {measured}")
        self._dimension = measured

    @property
    def dimension(self):
        return self._dimension

    @property
    def distance_metric(self):
        return self.spec.distance_metric

    def embed_documents(self, texts):
        return [list(x) for x in self._model.embed([self.spec.document_prefix + x for x in texts])]

    def embed_query(self, text):
        return list(next(iter(self._model.embed([self.spec.query_prefix + text]))))
