"""Hybrid retrieval over the local corpus: embedding similarity fused
with lexical token F1."""

from .corpus import PASSAGES
from .embed import cosine
from .textutil import token_f1

W_EMB = 0.7
W_LEX = 0.3


class Retriever:
    def __init__(self, embedder, passages=PASSAGES):
        self.passages = list(passages)
        self._vecs = {
            p["id"]: embedder.encode_one(p["text"]) for p in self.passages
        }
        self.embedder = embedder

    def search(self, query, k=3):
        qv = self.embedder.encode_one(query)
        scored = []
        for p in self.passages:
            s = W_EMB * cosine(qv, self._vecs[p["id"]]) + W_LEX * token_f1(
                query, p["text"]
            )
            scored.append((p["id"], s))
        scored.sort(key=lambda x: -x[1])
        return scored[:k]
