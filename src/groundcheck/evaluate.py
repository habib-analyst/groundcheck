"""Evaluation: measure the citation verifier and the full pipeline.

Part 1 — verifier classification on the planted ground-truth set
(dev split tunes the threshold; test split is reported):
  precision / recall / F1 for flagging unsupported citations,
  false-flag rate on supported citations, per-subtype catch rates,
  and ablations (embedding-only, lexical-only, combined).

Part 2 — end-to-end pipeline: noisy generated answers go through
verify -> fallback-repair/drop. Reports the unsupported-claim rate
before vs after, catch rate, repair accuracy, and answer-level
low-confidence flag quality.

Writes results.json. Run:
  GROUNDCHECK_MODEL=<local model dir> python -m groundcheck.evaluate
"""

import json
import os
from collections import defaultdict

from .corpus import BY_ID, PASSAGES
from .embed import Embedder
from .evalset import build_items, split_dev_test
from .generate import NoisyGenerator
from .pipeline import GroundCheckAgent
from .verify import SupportScorer


def classification_metrics(pairs):
    """pairs: list of (is_unsupported, flagged)."""
    tp = sum(1 for u, f in pairs if u and f)
    fp = sum(1 for u, f in pairs if not u and f)
    fn = sum(1 for u, f in pairs if u and not f)
    tn = sum(1 for u, f in pairs if not u and not f)
    prec = tp / (tp + fp) if tp + fp else 1.0
    rec = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return dict(
        precision=round(prec, 4),
        recall_catch_rate=round(rec, 4),
        f1=round(f1, 4),
        false_flag_rate=round(fp / (fp + tn), 4) if fp + tn else 0.0,
        accuracy=round((tp + tn) / max(len(pairs), 1), 4),
        tp=tp, fp=fp, fn=fn, tn=tn,
    )


def tune_threshold(scored_items, key):
    """scored_items: list of (item, SupportResult). Pick threshold that
    maximises F1 for flagging unsupported on these (dev) items."""
    best_tau, best_f1 = 0.55, -1.0
    for i in range(25, 91):
        tau = i / 100.0
        pairs = [
            (it["label"] == "unsupported", getattr(res, key) < tau)
            for it, res in scored_items
        ]
        m = classification_metrics(pairs)
        if m["f1"] > best_f1:
            best_f1, best_tau = m["f1"], tau
    return best_tau


def score_items(scorer, items):
    return [
        (it, scorer.features(it["claim"], it["cited"], BY_ID[it["cited"]]["text"]))
        for it in items
    ]


class _ScoredView:
    """Expose ablation scores through the same attribute interface."""

    def __init__(self, res):
        self.score = res.score
        self.emb = res.emb_sim
        from .verify import SupportScorer as _SS

        self.lex = _SS.lexical_only(res)


def main(out_path=None):
    embedder = Embedder()
    scorer = SupportScorer(embedder)
    items = build_items()
    dev, test = split_dev_test(items)
    dev_scored = [(_it, _ScoredView(_r)) for _it, _r in score_items(scorer, dev)]
    test_scored = [(_it, _ScoredView(_r)) for _it, _r in score_items(scorer, test)]

    results = dict(
        model=embedder.model_name,
        seed=20261026,
        eval_set=dict(
            total=len(items),
            dev=len(dev),
            test=len(test),
            test_supported=sum(1 for it in test if it["label"] == "supported"),
            test_unsupported=sum(1 for it in test if it["label"] == "unsupported"),
        ),
    )

    scorers = {}
    for name, key in [("combined", "score"), ("embedding_only", "emb"), ("lexical_only", "lex")]:
        tau = tune_threshold(dev_scored, key)
        pairs = [
            (it["label"] == "unsupported", getattr(res, key) < tau)
            for it, res in test_scored
        ]
        m = classification_metrics(pairs)
        m["threshold"] = tau
        scorers[name] = m
    results["verifier"] = scorers

    # per-subtype catch rate for the combined scorer
    tau = scorers["combined"]["threshold"]
    by_subtype = defaultdict(list)
    for it, res in test_scored:
        by_subtype[it["subtype"]].append(
            (it["label"] == "unsupported", res.score < tau)
        )
    results["catch_rate_by_subtype"] = {
        st: classification_metrics(ps)["recall_catch_rate"]
        for st, ps in sorted(by_subtype.items())
        if any(u for u, _ in ps)
    }

    # ---- end-to-end pipeline ------------------------------------------
    agent = GroundCheckAgent(embedder, threshold=tau, generator=NoisyGenerator(seed=7))
    answers = [agent.answer(p["question"], p["id"]) for p in PASSAGES]

    total_claims = bad_claims = caught = false_flags = clean_claims = 0
    repair_attempts = repair_success = 0
    final_total = final_unsupported = 0
    ans_pairs = []
    retrieval_hits = 0
    for ans, p in zip(answers, PASSAGES):
        if p["id"] in ans["retrieved"]:
            retrieval_hits += 1
        answer_has_bad = False
        for c in ans["claims"]:
            is_bad = c["truth"] in ("wrong_citation", "mutated_fact")
            total_claims += 1
            if is_bad:
                bad_claims += 1
                answer_has_bad = True
                if c["action"] in ("repaired", "dropped"):
                    caught += 1
                if c["truth"] == "wrong_citation" and c["action"] == "repaired":
                    repair_attempts += 1
                    if c["cited_final"] == c["true_passage"]:
                        repair_success += 1
            else:
                clean_claims += 1
                if c["action"] != "kept":
                    false_flags += 1
        for c in ans["final_claims"]:
            final_total += 1
            truly_supported = (
                c["truth"] != "mutated_fact" and c["cited_final"] == c["true_passage"]
            )
            if not truly_supported:
                final_unsupported += 1
        ans_pairs.append((answer_has_bad, ans["low_confidence"]))

    ans_m = classification_metrics(ans_pairs)
    results["end_to_end"] = dict(
        questions=len(answers),
        retrieval_recall_at_3=round(retrieval_hits / len(answers), 4),
        total_claims=total_claims,
        raw_unsupported_claims=bad_claims,
        raw_unsupported_rate=round(bad_claims / total_claims, 4) if total_claims else 0.0,
        unsupported_catch_rate=round(caught / bad_claims, 4) if bad_claims else 1.0,
        false_flag_rate_clean_claims=round(false_flags / clean_claims, 4) if clean_claims else 0.0,
        repair_success_rate=round(repair_success / repair_attempts, 4) if repair_attempts else None,
        final_claims=final_total,
        residual_unsupported_rate=round(final_unsupported / final_total, 4) if final_total else 0.0,
        answers_flagged_low_confidence=sum(1 for _, f in ans_pairs if f),
        answer_level_flag_precision=ans_m["precision"],
        answer_level_flag_recall=ans_m["recall_catch_rate"],
        abstentions=sum(1 for a in answers if a["abstained"]),
    )

    out_path = out_path or os.path.join(os.getcwd(), "results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
    print(f"\nwrote {out_path}")
    return results


if __name__ == "__main__":
    main()
