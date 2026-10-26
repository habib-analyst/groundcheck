"""Citation-support verification — the core of groundcheck.

Given a claim and the passage it cites, decide whether the passage
actually supports the claim. The scorer combines three local signals:

  * embedding similarity (all-MiniLM-L6-v2 cosine) — paraphrase-aware,
  * token F1 — lexical grounding of the claim in the passage,
  * number agreement — every number in the claim must appear in the
    passage; a missing number caps the score, because a claim whose
    figures are absent from its citation is the classic RAG failure
    (fluent text, invented fact).

No LLM judge, no API. Scores are deterministic.
"""

from dataclasses import dataclass

from .embed import cosine
from .textutil import number_agreement, token_f1

W_EMB = 0.55
W_LEX = 0.30
W_NUM = 0.15
NUMBER_MISMATCH_CAP = 0.40


@dataclass
class SupportResult:
    score: float
    emb_sim: float
    lex_f1: float
    num_agreement: float

    def supported(self, threshold):
        return self.score >= threshold


class SupportScorer:
    def __init__(self, embedder):
        self.embedder = embedder
        self._passage_vecs = {}

    def passage_vec(self, passage_id, passage_text):
        if passage_id not in self._passage_vecs:
            self._passage_vecs[passage_id] = self.embedder.encode_one(passage_text)
        return self._passage_vecs[passage_id]

    def features(self, claim, passage_id, passage_text):
        emb = cosine(
            self.embedder.encode_one(claim),
            self.passage_vec(passage_id, passage_text),
        )
        return self.features_from_emb(claim, passage_text, emb)

    @staticmethod
    def features_from_emb(claim, passage_text, emb_sim):
        lex = token_f1(claim, passage_text)
        num = number_agreement(claim, passage_text)
        score = W_EMB * emb_sim + W_LEX * lex + W_NUM * num
        if num == 0.0:
            score = min(score, NUMBER_MISMATCH_CAP)
        return SupportResult(score=score, emb_sim=emb_sim, lex_f1=lex, num_agreement=num)

    def score(self, claim, passage_id, passage_text):
        return self.features(claim, passage_id, passage_text).score

    # -- ablations ------------------------------------------------------
    @staticmethod
    def embedding_only(result):
        return result.emb_sim

    @staticmethod
    def lexical_only(result):
        score = 0.7 * result.lex_f1 + 0.3 * result.num_agreement
        if result.num_agreement == 0.0:
            score = min(score, NUMBER_MISMATCH_CAP)
        return score
