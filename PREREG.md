# groundcheck v2 — pre-registration

Written **before** any v2 benchmark run (2026-10-28). The bars and the
tuning protocol below are locked; the test split is touched once, for
reporting. The only model contact before this file was a load/smoke
check of the NLI weights on raw items (no thresholds fitted, no
metrics computed).

## What v2 changes

v1's scorer (embedding + token F1 + number agreement) has no negation
or entailment signal, and its measured blind spot is contradicted
claims: catch rate **0.286** on the test split (2/7). v2 adds one
stage: a **local NLI contradiction veto**.

- Model: `cross-encoder/nli-MiniLM2-L6-H768` (MiniLMv2-L6 distilled
  from RoBERTa-Large, fine-tuned for 3-way NLI), run locally on CPU
  via transformers. Premise = cited passage, hypothesis = claim.
  Signal = P(contradiction). Weights are **not** in this repo; they
  are a local model directory, same pattern as v1's MiniLM embeddings.
- v2 decision rule (verifier): flag a citation as unsupported iff

      v1_score < tau   OR   P_NLI(contradiction) >= tau_nli

- Pipeline: the same veto applies when a claim is verified against
  its cited passage, and a fallback repair is accepted only if the
  candidate passage does **not** trip the veto for that claim.

## Benchmark

Identical to v1: the planted ground-truth set from `evalset.py`
(204 items, seed 20261026), same stratified dev/test split
(dev 81 for tuning, test 123 for reporting: 29 supported /
94 unsupported, of which 7 contradicted). v1 numbers are recomputed
by the same run, side by side, from the same scored items — no
recycled v1 printouts.

## Tuning protocol (dev split only)

1. Tune the v1 threshold `tau` on dev exactly as v1 did
   (F1-maximising grid 0.25–0.90).
2. With `tau` fixed, grid `tau_nli` over
   {0.05, 0.10, …, 0.95}; choose the value maximising **dev F1** of
   the v2 rule. Ties break toward the **larger** `tau_nli`
   (fewer vetoes).

## Locked bars

**Primary bar (the v2 claim):**
- test contradiction catch rate **> 0.70** (v1: 0.286), AND
- test verifier F1 **≥ 0.95** (v1: 0.967).

**No-regression guard (all must hold to ship as an update):**
- test precision ≥ 0.95 (v1: 0.989),
- test overall recall ≥ 0.947 (v1 level; an OR-veto cannot lower it),
- end-to-end residual unsupported rate stays 0.0 (v1: 0.0),
- end-to-end false-flag rate on clean claims ≤ 0.05 (v1: 0.0;
  small tolerance pre-committed because the pipeline applies the
  same veto),
- full pytest suite passes.

## Decision rule

- Primary bar met **and** guard holds → v2 ships as an update.
- Primary bar missed but guard holds → the miss is documented and
  v2 may still ship as an update with honest numbers.
- Guard broken (anything regresses) → `KILLED.md`; v2 does not ship.
