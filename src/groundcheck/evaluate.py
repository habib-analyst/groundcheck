"""Evaluation: measure the citation verifier (v1 and v2) and the pipeline.

Part 1 — verifier classification on the planted ground-truth set
(dev split tunes thresholds; test split is reported):
  precision / recall / F1 for flagging unsupported citations,
  false-flag rate on supported citations, per-subtype catch rates,
  and ablations (embedding-only, lexical-only, combined).

  v2 = the v1 combined score plus a local NLI contradiction veto
  (PREREG.md): flag iff score < tau OR P_NLI(contradiction) >= tau_nli.
  tau_nli is tuned on the dev split only, by the locked protocol in
  PREREG.md. v1 numbers are recomputed in the same run from the same
  scored items — nothing is recycled from the v1 printout.

Part 2 — end-to-end pipeline (v1 agent and v2 agent on identical
planted-noise drafts): unsupported-claim rate before vs after, catch
rate, repair accuracy, answer-level low-confidence flag quality.

Writes results.json, including a `prereg_verdict` block that scores
the locked bars programmatically. Run:
  GROUNDCHECK_MODEL=<local model dir> \
  GROUNDCHECK_NLI_MODEL=<local NLI model dir> \
    python -m groundcheck.evaluate
Set GROUNDCHECK_NO_NLI=1 to reproduce the v1-only evaluation.
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

SEED = 20261026


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


def tune_nli_threshold(dev_scored, tau):
    """PREREG protocol: with the v1 threshold fixed, grid tau_nli over
    {0.05, ..., 0.95} and pick the value maximising dev F1 of the v2
    rule; ties break toward the larger tau_nli (fewer vetoes)."""
    best_tau_nli, best_f1 = None, -1.0
    for i in range(5, 100, 5):
        tau_nli = i / 100.0
        pairs = [
            (it["label"] == "unsupported", res.flagged_v2(tau, tau_nli))
            for it, res in dev_scored
        ]
        m = classification_metrics(pairs)
        if m["f1"] >= best_f1:  # >= : later (larger) tau_nli wins ties
            best_f1, best_tau_nli = m["f1"], tau_nli
    return best_tau_nli


def score_items(scorer, items):
    return [
        (it, scorer.features(it["claim"], it["cited"], BY_ID[it["cited"]]["text"]))
        for it in items
    ]


def catch_by_subtype(scored_items, flag_fn):
    by_subtype = defaultdict(list)
    for it, res in scored_items:
        by_subtype[it["subtype"]].append(
            (it["label"] == "unsupported", flag_fn(res))
        )
    return {
        st: classification_metrics(ps)["recall_catch_rate"]
        for st, ps in sorted(by_subtype.items())
        if any(u for u, _ in ps)
    }


class _ScoredView:
    """Expose ablation scores through the same attribute interface."""

    def __init__(self, res):
        self.score = res.score
        self.emb = res.emb_sim
        self.nli_contradiction = res.nli_contradiction
        from .verify import SupportScorer as _SS

        self.lex = _SS.lexical_only(res)

    def flagged_v2(self, tau, tau_nli):
        return self.score < tau or (
            tau_nli is not None
            and self.nli_contradiction is not None
            and self.nli_contradiction >= tau_nli
        )


def run_end_to_end(agent):
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
    return dict(
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


def main(out_path=None):
    embedder = Embedder()
    scorer = SupportScorer(embedder)
    items = build_items()
    dev, test = split_dev_test(items)

    use_nli = os.environ.get("GROUNDCHECK_NO_NLI") != "1"
    nli = None
    if use_nli:
        from .nli import NLIClassifier

        nli = NLIClassifier()

    dev_raw = score_items(scorer, dev)
    test_raw = score_items(scorer, test)
    if nli is not None:
        scorer.attach_nli(dev_raw, nli)
        scorer.attach_nli(test_raw, nli)
    dev_scored = [(_it, _ScoredView(_r)) for _it, _r in dev_raw]
    test_scored = [(_it, _ScoredView(_r)) for _it, _r in test_raw]

    results = dict(
        model=embedder.model_name,
        seed=SEED,
        eval_set=dict(
            total=len(items),
            dev=len(dev),
            test=len(test),
            test_supported=sum(1 for it in test if it["label"] == "supported"),
            test_unsupported=sum(1 for it in test if it["label"] == "unsupported"),
        ),
    )

    # ---- v1 verifier (recomputed) --------------------------------------
    scorers = {}
    for name, key in [("combined", "score"), ("embedding_only", "emb"), ("lexical_only", "lex")]:
        tau_k = tune_threshold(dev_scored, key)
        pairs = [
            (it["label"] == "unsupported", getattr(res, key) < tau_k)
            for it, res in test_scored
        ]
        m = classification_metrics(pairs)
        m["threshold"] = tau_k
        scorers[name] = m
    results["verifier"] = scorers

    tau = scorers["combined"]["threshold"]
    results["catch_rate_by_subtype"] = catch_by_subtype(
        test_scored, lambda res: res.score < tau
    )

    # ---- end-to-end v1 ---------------------------------------------------
    agent = GroundCheckAgent(embedder, threshold=tau, generator=NoisyGenerator(seed=7))
    results["end_to_end"] = run_end_to_end(agent)

    # ---- v2: NLI contradiction veto --------------------------------------
    if nli is not None:
        tau_nli = tune_nli_threshold(dev_scored, tau)
        pairs = [
            (it["label"] == "unsupported", res.flagged_v2(tau, tau_nli))
            for it, res in test_scored
        ]
        m2 = classification_metrics(pairs)
        m2["threshold"] = tau
        m2["nli_threshold"] = tau_nli
        results["nli_model"] = nli.model_name
        results["verifier_v2"] = m2
        results["catch_rate_by_subtype_v2"] = catch_by_subtype(
            test_scored, lambda res: res.flagged_v2(tau, tau_nli)
        )

        agent_v2 = GroundCheckAgent(
            embedder,
            threshold=tau,
            generator=NoisyGenerator(seed=7),
            nli=nli,
            nli_threshold=tau_nli,
        )
        results["end_to_end_v2"] = run_end_to_end(agent_v2)

        # ---- PREREG verdict (locked bars, PREREG.md) ---------------------
        v1m = scorers["combined"]
        e2 = results["end_to_end_v2"]
        contra_v2 = results["catch_rate_by_subtype_v2"].get("contradicted", 0.0)
        verdict = dict(
            contradiction_catch_v2=contra_v2,
            bar_contradiction_catch_gt_0_70=bool(contra_v2 > 0.70),
            bar_f1_ge_0_95=bool(m2["f1"] >= 0.95),
            guard_precision_ge_0_95=bool(m2["precision"] >= 0.95),
            guard_recall_ge_v1=bool(
                m2["recall_catch_rate"] >= v1m["recall_catch_rate"]
            ),
            guard_e2e_residual_zero=bool(e2["residual_unsupported_rate"] == 0.0),
            guard_e2e_false_flag_le_0_05=bool(
                e2["false_flag_rate_clean_claims"] <= 0.05
            ),
        )
        verdict["primary_bar_met"] = bool(
            verdict["bar_contradiction_catch_gt_0_70"] and verdict["bar_f1_ge_0_95"]
        )
        verdict["guard_holds"] = bool(
            all(v for k, v in verdict.items() if k.startswith("guard_"))
        )
        results["prereg_verdict"] = verdict

    out_path = out_path or os.path.join(os.getcwd(), "results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
    print(f"\nwrote {out_path}")
    return results


if __name__ == "__main__":
    main()
