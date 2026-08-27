import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
for folder in ("shared", "daemon", "client"):
    sys.path.insert(0, str(ROOT / folder))


class FakeEmbedder:
    model_id = "test"
    dimension = 3

    def embed_documents(self, texts):
        return [self.embed_query(x) for x in texts]

    def embed_query(self, text):
        text = text.lower()
        return [float("reactor" in text or "pressure" in text), float("storage" in text), float(len(text) % 7) / 7]
