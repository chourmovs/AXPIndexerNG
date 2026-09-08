class CPUEmbeddingBackend:
    """Adapter around the unchanged FastEmbed reference implementation."""

    backend = "fastembed"
    effective_device = "cpu"
    intel_gpu_qualified = False
    intel_gpu_device_name = None
    fallback_reason = None

    def __init__(self, model, *, cache_dir=None, batch_size=64, local_only=True):
        # Kept lazy to avoid an embeddings -> backends import cycle.
        from ..embeddings import Embedder

        self._embedder = Embedder(model, cache_dir=cache_dir, local_only=local_only,
                                  runtime_batch_size=batch_size)
        self.model_id = self._embedder.model_id
        self.dimension = self._embedder.dimension
        self.distance_metric = self._embedder.distance_metric

    def embed_documents(self, texts):
        return self._embedder.embed_documents(texts)

    def embed_query(self, text):
        return self._embedder.embed_query(text)

    def health(self):
        return {"backend": self.backend, "effective_device": self.effective_device, "healthy": True}
