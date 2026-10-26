"""Planted ground-truth evaluation set.

Every item is a (claim, cited passage) pair whose support label is known
*by construction* — no human or LLM annotation, so scoring is exact:

  supported      gold claim cited to the passage it came from
  wrong_passage  gold claim cited to a passage from a different cluster
  hard_negative  gold claim cited to its cluster-mate (same topic,
                 different facts — topically close, not supported)
  mutated_fact   gold claim with its first number perturbed, cited to
                 its true passage (the passage does not support it)
  contradicted   hand-written claim the cited passage contradicts

Split deterministically into dev (threshold tuning) and test (reporting),
stratified by subtype.
"""

import random

from .corpus import BY_ID, CONTRADICTED, PASSAGES, cluster_mate
from .textutil import mutate_first_number

SEED = 20261026


def build_items(seed=SEED):
    rng = random.Random(seed)
    items = []
    other_cluster = [
        p["id"] for p in PASSAGES
    ]
    for p in PASSAGES:
        pid = p["id"]
        for claim in p["claims"]:
            items.append(
                dict(claim=claim, cited=pid, label="supported", subtype="supported")
            )
            wrong_pool = [
                q["id"]
                for q in PASSAGES
                if q["cluster"] != p["cluster"]
            ]
            items.append(
                dict(
                    claim=claim,
                    cited=rng.choice(wrong_pool),
                    label="unsupported",
                    subtype="wrong_passage",
                )
            )
            items.append(
                dict(
                    claim=claim,
                    cited=cluster_mate(pid),
                    label="unsupported",
                    subtype="hard_negative",
                )
            )
            mutated = mutate_first_number(claim, avoid_text=p["text"])
            assert mutated is not None and mutated != claim
            items.append(
                dict(
                    claim=mutated,
                    cited=pid,
                    label="unsupported",
                    subtype="mutated_fact",
                )
            )
    for pid, claim in CONTRADICTED:
        items.append(
            dict(claim=claim, cited=pid, label="unsupported", subtype="contradicted")
        )
    # sanity: every item's cited passage exists; supported items' claim
    # numbers must all appear in the cited passage (checked in tests too)
    for it in items:
        assert it["cited"] in BY_ID
    rng.shuffle(items)
    return items


def split_dev_test(items, dev_frac=0.4, seed=SEED):
    rng = random.Random(seed + 1)
    by_subtype = {}
    for it in items:
        by_subtype.setdefault(it["subtype"], []).append(it)
    dev, test = [], []
    for subtype, group in sorted(by_subtype.items()):
        group = list(group)
        rng.shuffle(group)
        n_dev = round(len(group) * dev_frac)
        dev.extend(group[:n_dev])
        test.extend(group[n_dev:])
    rng.shuffle(dev)
    rng.shuffle(test)
    return dev, test
