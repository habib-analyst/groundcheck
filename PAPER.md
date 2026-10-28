# groundcheck v2: an NLI contradiction veto for citation verification

## Abstract

groundcheck v1 verifies RAG citations with a local fusion of embedding
similarity, token F1 and number agreement, and its paper reported the
failure mode plainly: **contradicted claims were caught only 28.6% of
the time**, because a claim that negates its passage keeps the same
words and numbers, so every similarity signal scores it as supported.
v2 adds exactly one stage aimed at that hole: a local NLI model
(cross-encoder/nli-MiniLM2-L6-H768, run on CPU, no API) whose
P(contradiction) for (passage, claim) acts as a **veto** — a citation
is flagged if the v1 score fails *or* the veto fires. On the same
planted ground-truth benchmark (204 items, identical split, v1
numbers recomputed in the same run), the veto raises the contradicted
catch rate from **0.286 to 1.000** (7/7 test items) and overall recall
from 0.947 to 1.000, at a small precision cost (0.989 → 0.979: the
veto false-fires on one supported claim). F1 improves 0.967 → 0.990.
End-to-end, the residual unsupported rate stays 0.0% and repair
accuracy stays 100%, with one clean claim falsely flagged (4.8%).
Bars and tuning protocol were pre-registered in PREREG.md before the
run; both the primary bar and the no-regression guard passed.

## 1. What changed (and what did not)

**Unchanged:** the corpus, the planted evaluation set and split, the
v1 support score and its fusion weights, the fallback repair/drop
pipeline, the generator and its noise rates. v1 metrics in this paper
were recomputed by the v2 run and reproduce v1's published numbers
exactly (P 0.989 / R 0.947 / F1 0.967, contradicted 0.286).

**Added:** for each (claim, cited passage) pair, an NLI forward pass
with premise = passage, hypothesis = claim. Decision rule:

    flag as unsupported  iff  v1_score < tau  OR  P(contradiction) >= tau_nli

The veto is deliberately *not* blended into the score: a weighted sum
would let high similarity outvote a contradiction, which is precisely
the v1 failure. In the pipeline, the same veto gates both claim
verification and fallback-repair acceptance (a repair may not point a
claim at a passage that contradicts it).

**Thresholds.** tau = 0.52 tuned on dev exactly as v1. tau_nli = 0.65
tuned on the dev split only, by the locked PREREG protocol
(F1-maximising grid 0.05–0.95, ties toward larger tau_nli).

## 2. Results

Verifier, test split (n=123; 29 supported / 94 unsupported; positive
class = unsupported):

| Scorer | Precision | Recall | F1 | False-flag |
|---|---|---|---|---|
| v1 combined | 0.989 | 0.947 | 0.967 | 3.4% |
| **v2 (v1 + NLI veto)** | **0.979** | **1.000** | **0.990** | **6.9%** |
| v1 embedding-only | 0.785 | 0.968 | 0.867 | 86.2% |
| v1 lexical-only | 1.000 | 0.957 | 0.978 | 0.0% |

Catch rate by subtype:

| Subtype | v1 | v2 |
|---|---|---|
| wrong_passage | 1.000 | 1.000 |
| hard_negative | 1.000 | 1.000 |
| mutated_fact | 1.000 | 1.000 |
| **contradicted** | **0.286** | **1.000** |

End-to-end (24 questions, 48 claims, identical planted noise):

| Metric | v1 | v2 |
|---|---|---|
| Raw unsupported rate | 56.3% | 56.3% |
| Unsupported caught | 100% | 100% |
| Repair success | 100% | 100% |
| Residual unsupported in final answers | 0.0% | 0.0% |
| Clean claims falsely flagged | 0.0% (0/21) | 4.8% (1/21) |
| Final surviving claims | 39 | 38 |
| Answer-level flags: precision / recall | 0.80 / 1.00 | 0.76 / 1.00 |
| Abstentions | 2 | 2 |

**PREREG verdict:** primary bar met (contradiction catch 1.000 >
0.70; F1 0.990 ≥ 0.95); no-regression guard holds (precision 0.979 ≥
0.95; recall ≥ v1; residual 0.0; false-flag 0.048 ≤ 0.05).

## 3. Item-level honesty

The aggregate hides three facts worth stating:

- **The veto caught exactly the items v1 missed, on this set.** All 5
  contradicted test items that passed v1's score (0.53–0.64) have
  P(contradiction) 0.91–1.00. The two signals are complementary here,
  not redundant.
- **The NLI model has its own blind spot, and v1 covered it.** "Information
  can travel faster than the speed of light, according to relativity"
  gets P(contradiction) = 0.037 — the model reads the attribution and
  calls it entailment. That item is caught by the v1 score (0.495 <
  0.52), not the veto. Likewise the vague-quantifier contradiction
  ("the brain consumes only a negligible share…", P(contradiction) =
  0.237) is invisible to the veto. An NLI stage is not a negation
  oracle; it is one more fallible signal.
- **The veto's cost is real and specific.** Its one false fire on the
  test split is a true claim — "Tim Berners-Lee invented the World
  Wide Web in 1989 at CERN" — scored P(contradiction) = 0.974 against
  the passage that states exactly that. That single error is the whole
  precision dip (0.989 → 0.979), the end-to-end false flag (1/21 clean
  claims), and one surviving claim fewer in final answers (38 vs 39).
  On a larger corpus this rate, not the headline F1, is what would
  need watching.

## 4. Limitations (carried over and new)

- Same toy regime as v1: 24 self-authored passages, extractive-style
  gold claims, template generator. 7 contradicted test items is a
  small denominator — 1.000 means 7/7, and the honest confidence
  interval around it is wide. Nothing here measures paraphrase-heavy
  or LLM-generated claims.
- The NLI model adds ~330 MB of weights and real latency (a full
  benchmark run is minutes on a contended 2-core VM; per-claim
  pipeline checks are single forward passes). v1's "no model beyond
  embeddings" simplicity is gone; that is the price of the entailment
  signal, and it is paid knowingly.
- Contradiction is judged against the whole passage as one premise.
  Long real-world passages with mixed support would need
  sentence-level NLI; unmeasured here.
- Thresholds (tau, tau_nli) are fitted to this corpus family on an
  81-item dev split; transfer is unmeasured, same caveat as v1.

## 5. Reproduction

```bash
pip install -e .
# local all-MiniLM-L6-v2 dir + local NLI dir (see README for the
# curl commands that fetch the NLI weights):
GROUNDCHECK_MODEL=<all-MiniLM-L6-v2 dir> \
GROUNDCHECK_NLI_MODEL=<nli-MiniLM2-L6-H768 dir> \
  PYTHONPATH=src python -m groundcheck.evaluate   # writes results.json
pytest                              # 22 tests
```

Deterministic: fixed seeds (eval set 20261026, generator 7),
CPU-only, no network calls at evaluation time. `results.json` in this
repository was produced by exactly this command; PREREG.md records
the bars that were locked before it ran.
