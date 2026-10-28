"""GroundCheck agent: retrieve -> generate -> verify -> repair/abstain.

Every claim in a generated answer is checked against the passage it
cites. A claim whose citation fails verification triggers the fallback
path: a corpus-wide re-search for a passage that actually supports the
claim. Found -> the citation is repaired. Not found -> the claim is
dropped and the answer is flagged low-confidence; if no claims survive,
the agent abstains instead of answering.

v2: when an NLI model and threshold are supplied, the NLI
contradiction veto also applies — a claim whose cited passage
contradicts it fails verification even if the v1 score passes, and a
fallback repair is accepted only if the candidate passage does not
trip the veto for that claim. With no NLI supplied, behaviour is
exactly v1.
"""

from .corpus import BY_ID, PASSAGES
from .generate import NoisyGenerator
from .retrieve import Retriever
from .verify import SupportScorer

ANSWER_CONFIDENCE_FLOOR = 0.65


class GroundCheckAgent:
    def __init__(self, embedder, threshold=0.55, generator=None,
                 nli=None, nli_threshold=None):
        self.embedder = embedder
        self.scorer = SupportScorer(embedder, nli=nli)
        self.retriever = Retriever(embedder)
        self.threshold = threshold
        self.generator = generator or NoisyGenerator()
        self.nli_threshold = nli_threshold if nli is not None else None

    # -- verification + fallback ---------------------------------------
    def verify_claim(self, claim):
        cited_text = BY_ID[claim["cited"]]["text"]
        res = self.scorer.features(claim["text"], claim["cited"], cited_text)
        if self.nli_threshold is not None:
            res.nli_contradiction = self.scorer.nli_contradiction(
                claim["text"], cited_text
            )
        return res

    def _vetoed(self, claim_text, passage_text):
        """NLI veto check for one (passage, claim) pair (v2 only)."""
        if self.nli_threshold is None:
            return False
        p_con = self.scorer.nli_contradiction(claim_text, passage_text)
        return p_con is not None and p_con >= self.nli_threshold

    def fallback_search(self, claim_text):
        """Score the claim against every passage; return (best_id, score)."""
        best_id, best_score = None, -1.0
        for p in PASSAGES:
            s = self.scorer.score(claim_text, p["id"], p["text"])
            if s > best_score:
                best_id, best_score = p["id"], s
        return best_id, best_score

    def process_claim(self, claim):
        """Verify one claim; repair or drop it. Returns an outcome record."""
        res = self.verify_claim(claim)
        out = dict(
            text=claim["text"],
            cited_original=claim["cited"],
            cited_final=claim["cited"],
            true_passage=claim["true_passage"],
            truth=claim["truth"],
            score_initial=res.score,
            nli_contradiction=res.nli_contradiction,
            vetoed=res.vetoed(self.nli_threshold),
            action="kept",
            repaired=False,
        )
        if res.supported(self.threshold) and not out["vetoed"]:
            out["score_final"] = res.score
            return out
        # fallback: search the whole corpus for real support
        best_id, best_score = self.fallback_search(claim["text"])
        if (
            best_score >= self.threshold
            and best_id != claim["cited"]
            and not self._vetoed(claim["text"], BY_ID[best_id]["text"])
        ):
            out["cited_final"] = best_id
            out["score_final"] = best_score
            out["action"] = "repaired"
            out["repaired"] = True
            return out
        out["action"] = "dropped"
        out["score_final"] = None
        return out

    # -- end to end ------------------------------------------------------
    def answer(self, question, target_passage_id):
        retrieved = self.retriever.search(question, k=3)
        retrieved_ids = [pid for pid, _ in retrieved]
        draft = self.generator.generate(question, target_passage_id, retrieved_ids)
        processed = [self.process_claim(c) for c in draft["claims"]]
        final_claims = [p for p in processed if p["action"] != "dropped"]
        dropped = [p for p in processed if p["action"] == "dropped"]
        repaired = [p for p in processed if p["repaired"]]
        flagged = [p for p in processed if p["action"] != "kept"]
        if final_claims:
            confidence = sum(p["score_final"] for p in final_claims) / len(
                final_claims
            )
        else:
            confidence = 0.0
        low_confidence = bool(
            dropped or repaired or confidence < ANSWER_CONFIDENCE_FLOOR
        )
        return dict(
            question=question,
            retrieved=retrieved_ids,
            claims=processed,
            final_claims=final_claims,
            n_dropped=len(dropped),
            n_repaired=len(repaired),
            n_flagged=len(flagged),
            confidence=confidence,
            low_confidence=low_confidence,
            abstained=len(final_claims) == 0,
        )
