import pytest

from groundcheck.corpus import BY_ID, CLUSTERS, PASSAGES
from groundcheck.evalset import build_items, split_dev_test
from groundcheck.textutil import (
    extract_numbers,
    mutate_first_number,
    number_agreement,
    token_f1,
)


# ---- corpus & eval-set construction (no model needed) -------------------

def test_corpus_integrity():
    assert len(PASSAGES) == 24
    assert len({p["id"] for p in PASSAGES}) == 24
    assert all(len(p["claims"]) == 2 for p in PASSAGES)
    assert all(len(ids) == 2 for ids in CLUSTERS.values())


def test_gold_claims_are_supported_by_construction():
    for p in PASSAGES:
        for claim in p["claims"]:
            assert number_agreement(claim, p["text"]) == 1.0


def test_evalset_counts_and_labels():
    items = build_items()
    by_subtype = {}
    for it in items:
        by_subtype[it["subtype"]] = by_subtype.get(it["subtype"], 0) + 1
    assert by_subtype == {
        "supported": 48,
        "wrong_passage": 48,
        "hard_negative": 48,
        "mutated_fact": 48,
        "contradicted": 12,
    }
    assert sum(1 for it in items if it["label"] == "supported") == 48
    assert sum(1 for it in items if it["label"] == "unsupported") == 156


def test_evalset_deterministic():
    assert build_items() == build_items()


def test_split_partitions_all_items():
    items = build_items()
    dev, test = split_dev_test(items)
    assert len(dev) + len(test) == len(items)
    key = lambda it: (it["claim"], it["cited"], it["subtype"])
    assert not ({key(i) for i in dev} & {key(i) for i in test})


def test_mutated_items_really_unsupported():
    for it in build_items():
        if it["subtype"] == "mutated_fact":
            assert number_agreement(it["claim"], BY_ID[it["cited"]]["text"]) == 0.0


# ---- text utilities -------------------------------------------------------

def test_textutil_numbers():
    assert extract_numbers("about 10,935 metres and 2.5 cm") == [10935.0, 2.5]
    assert number_agreement("It is 30 metres long.", "reaching 30 metres") == 1.0
    assert number_agreement("It is 31 metres long.", "reaching 30 metres") == 0.0
    assert number_agreement("No figures here.", "reaching 30 metres") == 0.5


def test_token_f1_bounds():
    assert token_f1("the cat sat", "the cat sat") > 0.9
    assert token_f1("quantum bananas oscillate", "the cat sat") == 0.0


def test_mutate_first_number():
    out = mutate_first_number("The tower is 100 metres tall.", avoid_text="100 metres")
    assert out is not None and "100" not in out.split("is")[1].split("metres")[0]
    assert mutate_first_number("No numbers at all.") is None


# ---- model-backed behaviour -------------------------------------------------

@pytest.fixture(scope="session")
def embedder():
    from groundcheck.embed import Embedder

    return Embedder()


@pytest.fixture(scope="session")
def scorer(embedder):
    from groundcheck.verify import SupportScorer

    return SupportScorer(embedder)


def test_supported_scores_above_wrong_passage(scorer):
    p = BY_ID["P01"]
    claim = p["claims"][0]
    good = scorer.score(claim, "P01", p["text"])
    bad = scorer.score(claim, "P03", BY_ID["P03"]["text"])
    assert good > bad
    assert good >= 0.55


def test_mutated_claim_is_capped(scorer):
    p = BY_ID["P01"]
    mutated = mutate_first_number(p["claims"][0], avoid_text=p["text"])
    res = scorer.features(mutated, "P01", p["text"])
    assert res.num_agreement == 0.0
    assert res.score <= 0.40


def test_retrieval_finds_target(embedder):
    from groundcheck.retrieve import Retriever

    r = Retriever(embedder)
    for pid in ("P01", "P10", "P21"):
        top = [x for x, _ in r.search(BY_ID[pid]["question"], k=3)]
        assert pid in top


@pytest.fixture(scope="session")
def agent(embedder):
    from groundcheck.pipeline import GroundCheckAgent

    return GroundCheckAgent(embedder, threshold=0.55)


def _claim(text, cited, true):
    return dict(text=text, cited=cited, true_passage=true, truth="manual")


def test_pipeline_keeps_clean_claim(agent):
    p = BY_ID["P01"]
    out = agent.process_claim(_claim(p["claims"][0], "P01", "P01"))
    assert out["action"] == "kept"


def test_pipeline_repairs_wrong_citation(agent):
    p = BY_ID["P01"]
    out = agent.process_claim(_claim(p["claims"][0], "P03", "P01"))
    assert out["action"] == "repaired"
    assert out["cited_final"] == "P01"


def test_pipeline_drops_mutated_claim(agent):
    p = BY_ID["P01"]
    mutated = mutate_first_number(p["claims"][0], avoid_text=p["text"])
    out = agent.process_claim(_claim(mutated, "P01", "P01"))
    assert out["action"] == "dropped"


def test_fallback_finds_nothing_for_alien_claim(agent):
    best_id, best_score = agent.fallback_search(
        "The stock index rose by 12 percent in 1999 after the central bank cut rates."
    )
    assert best_score < agent.threshold


def test_agent_answer_structure(agent):
    ans = agent.answer(BY_ID["P05"]["question"], "P05")
    assert len(ans["claims"]) == 2
    assert isinstance(ans["low_confidence"], bool)
    assert 0.0 <= ans["confidence"] <= 1.0


# ---- v2: NLI contradiction veto ------------------------------------------

class StubNLI:
    """Deterministic NLI stand-in: flags pairs whose hypothesis carries
    a marker word, so veto logic can be tested without the real model."""

    def __init__(self, marker="CONTRADICTS"):
        self.marker = marker

    def contradiction_probs(self, pairs):
        return [0.99 if self.marker in h else 0.01 for _, h in pairs]


def test_veto_logic_on_support_result():
    from groundcheck.verify import SupportResult

    res = SupportResult(score=0.9, emb_sim=0.9, lex_f1=0.9, num_agreement=1.0)
    assert not res.vetoed(0.5)  # no NLI attached -> never vetoed
    assert not res.flagged_v2(0.52, 0.5)
    res.nli_contradiction = 0.9
    assert res.vetoed(0.5)
    assert res.flagged_v2(0.52, 0.5)  # passes v1 score, vetoed by NLI
    res.nli_contradiction = 0.2
    assert not res.flagged_v2(0.52, 0.5)
    res.score = 0.3
    assert res.flagged_v2(0.52, 0.5)  # v1 failure still flags


@pytest.fixture(scope="session")
def agent_v2(embedder):
    from groundcheck.pipeline import GroundCheckAgent

    return GroundCheckAgent(
        embedder, threshold=0.55, nli=StubNLI(), nli_threshold=0.5
    )


def test_pipeline_v2_drops_vetoed_claim(agent_v2):
    p = BY_ID["P15"]
    claim = "CONTRADICTS: " + p["claims"][0]  # stub vetoes this pair
    out = agent_v2.process_claim(_claim(claim, "P15", "P15"))
    assert out["vetoed"] is True
    assert out["action"] == "dropped"  # no non-vetoed passage supports it


def test_pipeline_v2_keeps_clean_claim(agent_v2):
    p = BY_ID["P01"]
    out = agent_v2.process_claim(_claim(p["claims"][0], "P01", "P01"))
    assert out["vetoed"] is False
    assert out["action"] == "kept"


def test_attach_nli_batch_fills_results(embedder):
    from groundcheck.verify import SupportScorer

    scorer = SupportScorer(embedder)
    items = build_items()[:6]
    scored = [
        (it, scorer.features(it["claim"], it["cited"], BY_ID[it["cited"]]["text"]))
        for it in items
    ]
    assert all(r.nli_contradiction is None for _, r in scored)
    scorer.attach_nli(scored, StubNLI())
    assert all(r.nli_contradiction == 0.01 for _, r in scored)


def test_real_nli_model_if_available(embedder):
    """Runs only when GROUNDCHECK_NLI_MODEL points at a local model dir."""
    import os

    model_dir = os.environ.get("GROUNDCHECK_NLI_MODEL", "")
    if not model_dir or not os.path.isdir(model_dir):
        pytest.skip("no local NLI model configured")
    from groundcheck.corpus import CONTRADICTED
    from groundcheck.nli import NLIClassifier

    nli = NLIClassifier(model_dir)
    probs = nli.probs(BY_ID["P04"]["text"], CONTRADICTED[1][1])
    assert probs["contradiction"] > 0.9
    gold = BY_ID["P04"]["claims"][0]
    probs = nli.probs(BY_ID["P04"]["text"], gold)
    assert probs["contradiction"] < 0.1
