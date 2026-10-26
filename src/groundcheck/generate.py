"""A controlled answer generator with planted citation noise.

This stands in for an LLM generator so the *verification* layer can be
measured exactly: the generator composes an answer from gold claims of a
source passage, then corrupts it the way real RAG systems fail —

  * wrong_citation: the claim is true but cites the wrong retrieved
    passage (the citation does not support it as cited),
  * mutated_fact: the claim's number is perturbed while the citation
    stays correct (no passage supports the claim as stated).

Ground truth for every generated claim is tracked, so the pipeline's
catch rate, repair accuracy and residual error are exact, not judged.
"""

import random

from .corpus import BY_ID
from .textutil import mutate_first_number

P_WRONG_CITATION = 0.30
P_MUTATED = 0.15


class NoisyGenerator:
    def __init__(self, seed=7):
        self.rng = random.Random(seed)

    def generate(self, question, target_passage_id, retrieved_ids):
        """Return an answer dict with claims and per-claim ground truth."""
        if target_passage_id in retrieved_ids:
            source_id = target_passage_id
        else:
            source_id = retrieved_ids[0]
        source = BY_ID[source_id]
        claims = []
        for text in source["claims"]:
            truth = "clean"
            cited = source_id
            claim_text = text
            r = self.rng.random()
            if r < P_WRONG_CITATION:
                alts = [x for x in retrieved_ids if x != source_id]
                if alts:
                    cited = self.rng.choice(alts)
                    truth = "wrong_citation"
            elif r < P_WRONG_CITATION + P_MUTATED:
                mutated = mutate_first_number(
                    text, avoid_text=source["text"]
                )
                if mutated is not None:
                    claim_text = mutated
                    truth = "mutated_fact"
            claims.append(
                dict(
                    text=claim_text,
                    cited=cited,
                    true_passage=source_id,
                    truth=truth,
                )
            )
        return dict(question=question, claims=claims)
