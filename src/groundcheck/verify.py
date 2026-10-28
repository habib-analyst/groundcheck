"""Citation-support verification — the core of groundcheck.

Given a claim and the passage it cites, decide whether the passage
actually supports the claim. The v1 scorer combines three local
signals:

  * embedding similarity (all-MiniLM-L6-v2 cosine) — paraphrase-aware,
  * token F1 — lexical grounding of the claim in the passage,
  * number agreement — every number in the claim must appear in the
    passage; a missing number caps the score, because a claim whose
    figures are absent from its citation is the classic RAG failure
    (fluent text, invented fact).

v2 adds a fourth signal as a *veto*, not a blend: a local NLI model's
P(contradiction) for (passage, claim). Similarity signals cannot see
negation — a contradicted claim reuses its passage's words and
numbers — so the NLI stage runs alongside the v1 score and can flag a
citation the v1 score alone would pass. See PREREG.md for the locked
evaluation bars and `nli.py` for the model wrapper.

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
    # v2: local NLI P(contradiction) for (passage, claim); None when no
    # NLI model is attached (pure v1 behaviour).
    nli_contradiction: float = None

    def supported(self, threshold):
        return self.score >= threshold

    def vetoed(self, nli_threshold):
        """True when the NLI stage flags this pair as a contradiction."""
        return (
            nli_threshold is not None
            and self.nli_contradiction is not None
            and self.nli_contradiction >= nli_threshold
        )

    def flagged_v2(self, threshold, nli_threshold):
        """The v2 decision rule: unsupported iff the v1 score fails OR
        the NLI contradiction veto fires (see PREREG.md)."""
        return self.score < threshold or self.vetoed(nli_threshold)


class SupportScorer:
    def __init__(self, embedder, nli=None):
        self.embedder = embedder
        self.nli = nli
        self._passage_vecs = {}
        self._nli_cache = {}

    def passage_vec(self, passage_id, passage_text):
        if passage_id not in self._passage_vecs:
            self._passage_vecs[passage_id] = self.embedder.encode_one(passage_text)
        return self._passage_vecs[passage_id]

    def features(self, claim, passage_id, passage_text):
        """v1 features only — pure and cheap. NLI is attached explicitly
        (attach_nli for batches, nli_contradiction for single pairs) so
        the pipeline's corpus-wide fallback search never pays for NLI
        calls on passages it is only ranking."""
        emb = cosine(
            self.embedder.encode_one(claim),
            self.passage_vec(passage_id, passage_text),
        )
        return self.features_from_emb(claim, passage_text, emb)

    # -- NLI (v2) ---------------------------------------------------------
    def nli_contradiction(self, claim, passage_text):
        """P(contradiction) for one (passage, claim) pair, cached.
        Requires an NLI model to be attached."""
        if self.nli is None:
            return None
        key = (claim, passage_text)
        if key not in self._nli_cache:
            self._nli_cache[key] = self.nli.contradiction_probs(
                [(passage_text, claim)]
            )[0]
        return self._nli_cache[key]

    def attach_nli(self, scored, nli=None):
        """Batch-fill nli_contradiction on (item, SupportResult) pairs
        produced by score_items — one batched pass instead of a model
        call per item. Uses the given NLI classifier, or the scorer's
        own when attached. Returns the same list, mutated."""
        nli = nli or self.nli
        if nli is None:
            return scored
        from .corpus import BY_ID

        pairs = [(BY_ID[it["cited"]]["text"], it["claim"]) for it, _ in scored]
        for (_, res), (passage_text, claim), p_con in zip(
            scored, pairs, nli.contradiction_probs(pairs)
        ):
            res.nli_contradiction = p_con
            self._nli_cache[(claim, passage_text)] = p_con
        return scored

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
