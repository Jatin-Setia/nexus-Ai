import os
from fastembed import TextEmbedding

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class Embedder:
    """Drop-in replacement for SentenceTransformer("all-MiniLM-L6-v2").
    Uses fastembed (ONNX) instead of PyTorch to save RAM."""

    def __init__(self, name="sentence-transformers/all-MiniLM-L6-v2"):
        self._model = TextEmbedding(
            model_name=name,
            cache_dir=os.path.join(BASE_DIR, ".fastembed_cache"),
        )

    def encode(self, text):
        return next(iter(self._model.embed([text])))