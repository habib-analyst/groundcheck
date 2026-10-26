"""Local embedding wrapper around all-MiniLM-L6-v2 (sentence-transformers).

Fully local: point GROUNDCHECK_MODEL at a local model directory (or an HF
model id). Embeddings are L2-normalised; similarity is a dot product.
"""

import os

import numpy as np

DEFAULT_MODEL = os.environ.get(
    "GROUNDCHECK_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)


class Embedder:
    def __init__(self, model_name=DEFAULT_MODEL):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self._model = SentenceTransformer(model_name)

    def encode(self, texts):
        vecs = self._model.encode(
            list(texts), normalize_embeddings=True, show_progress_bar=False
        )
        return np.asarray(vecs, dtype=np.float64)

    def encode_one(self, text):
        return self.encode([text])[0]


def cosine(a, b):
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)
