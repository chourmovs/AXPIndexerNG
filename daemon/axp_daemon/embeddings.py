import os

FASTEMBED_MODEL = os.getenv("FASTEMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")


class Embedder:
    def __init__(self, model_id=FASTEMBED_MODEL, cache_dir=None, local_only=True):
        from fastembed import TextEmbedding

        self.model_id = model_id
        kwargs = {"model_name": model_id}
        if cache_dir:
            kwargs["cache_dir"] = cache_dir
        if local_only:
            kwargs["local_files_only"] = True
        self._model = TextEmbedding(**kwargs)
        self._dimension = len(next(iter(self._model.embed(["dimension probe"]))))

    @property
    def dimension(self):
        return self._dimension

    def embed_documents(self, texts):
        return [list(x) for x in self._model.embed(texts)]

    def embed_query(self, text):
        return self.embed_documents([text])[0]
