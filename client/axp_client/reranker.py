import hashlib
from collections import OrderedDict


RERANKER_MODEL = "answerdotai/answerai-colbert-small-v1"


def maxsim(query_embedding, document_embedding):
    """ColBERT MaxSim: sum, for every query token, its best document dot product."""
    return sum(max(sum(a * b for a, b in zip(q, d)) for d in document_embedding) for q in query_embedding)


class EmbeddingLRU:
    def __init__(self, capacity=128):
        self.capacity = max(1, int(capacity))
        self._values = OrderedDict()

    def key(self, chunk_id, text, model_id):
        return chunk_id, hashlib.sha256(text.encode()).hexdigest(), model_id

    def get(self, key):
        value = self._values.pop(key, None)
        if value is not None:
            self._values[key] = value
        return value

    def put(self, key, value):
        self._values.pop(key, None)
        self._values[key] = value
        while len(self._values) > self.capacity:
            self._values.popitem(last=False)


class Reranker:
    def __init__(self, cache_dir=None, local_only=True, cache_size=128, model_id=RERANKER_MODEL):
        try:
            from fastembed import LateInteractionTextEmbedding

            self.model = LateInteractionTextEmbedding(
                model_name=model_id, cache_dir=cache_dir, local_files_only=local_only
            )
        except Exception as exc:
            raise RuntimeError("Quality reranker model is not provisioned.") from exc
        self.model_id = model_id
        self.cache = EmbeddingLRU(cache_size)

    def score(self, query, candidates):
        query_embedding = next(iter(self.model.query_embed(query)))
        missing, keys = [], []
        embeddings = {}
        for item in candidates:
            key = self.cache.key(item["chunk_id"], item["snippet"], self.model_id)
            keys.append(key)
            cached = self.cache.get(key)
            if cached is None:
                missing.append(item)
            else:
                embeddings[item["chunk_id"]] = cached
        # One batched model invocation for every cache miss.
        if missing:
            generated = self.model.embed([item["snippet"] for item in missing])
            for item, value in zip(missing, generated):
                embeddings[item["chunk_id"]] = value
                self.cache.put(self.cache.key(item["chunk_id"], item["snippet"], self.model_id), value)
        return [maxsim(query_embedding, embeddings[item["chunk_id"]]) for item in candidates]
