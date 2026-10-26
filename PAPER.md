# groundcheck: citation-support verification inside the RAG loop

## Abstract

RAG answers carry citations, but citations are rarely checked: a cited
passage may be off-topic, on-topic but about different facts, or silent
about the numbers the sentence asserts. groundcheck adds a verification
stage inside the agent loop. Each (claim, cited passage) pair is scored
by a local, deterministic fusion of embedding similarity
(all-MiniLM-L6-v2), token F1, and a number-agreement rule; failures
trigger a corpus-wide fallback search that repairs the citation or
drops the claim, and answers are flagged low-confidence accordingly.
On a planted ground-truth set of 204 (claim, citation) pairs with exact
labels, the combined verifier flags unsupported citations with
precision 0.989 and recall 0.947 (test split, n=123). End-to-end, a
noisy generator's 56.3% raw unsupported-claim rate is reduced to a 0.0%
residual rate on a 24-question run, with no clean claim falsely
flagged. The clear failure mode is contradicted claims (28.6% caught):
similarity is not entailment, and this paper reports that plainly.

## 1. Method

**Support score.** For claim *c* and cited passage *p*:

    score = 0.55 · cos(MiniLM(c), MiniLM(p)) + 0.30 · tokenF1(c, p)
          + 0.15 · numAgreement(c, p)

where numAgreement is 1 if every number in *c* appears in *p*, 0 if
any is missing, 0.5 if *c* has no numbers. If numAgreement = 0 the
score is capped at 0.40: a claim whose figures are absent from its
citation cannot be rescued by topical similarity.

**Decision + fallback.** score ≥ τ (τ tuned on dev) → citation stands.
Otherwise the claim is scored against every passage in the corpus; if
the best passage clears τ and differs from the cited one, the citation
is replaced (repair). If nothing clears τ, the claim is dropped. An
answer is low-confidence if any claim was repaired or dropped or its
mean final score < 0.65; an answer with no surviving claims becomes an
abstention.

**Why these signals.** Embedding similarity alone cannot separate
support from topical relatedness (measured below: 86.2% false-flag rate
at its best threshold — it fires on nearly everything or nothing).
Lexical overlap grounds the claim's wording; the number rule targets
the highest-harm failure, invented quantities, which both similarity
signals are blind to because a mutated claim differs from a supported
one by a single token.

## 2. Evaluation design

**Corpus.** 24 short encyclopedia-style passages in 12 topic clusters
of two related passages with disjoint facts (e.g. Mariana Trench vs
Mid-Atlantic Ridge). Cluster-mates create hard negatives: same topic,
wrong facts. Each passage has two gold claims it supports.

**Planted set (204 items).** For each gold claim: (a) cited to its own
passage — *supported*; (b) cited to a random other-cluster passage —
*wrong_passage*; (c) cited to its cluster-mate — *hard_negative*;
(d) first number perturbed ×1.37 and cited to its true passage —
*mutated_fact*. Plus 12 hand-written *contradicted* claims. Labels are
known by construction; no annotation or LLM judging is involved.
Stratified split: dev 81 (threshold tuning only), test 123 (29
supported / 94 unsupported) for all reported numbers.

**End-to-end.** A template generator composes answers from gold claims
and plants noise (30% wrong citations, 15% number mutations). 24
questions, 48 claims, ground truth tracked per claim. A claim counts
as truly supported in the final answer iff it is unmutated and its
final citation is its source passage.

## 3. Results

Verifier, test split (positive class = unsupported):

| Scorer | τ | Precision | Recall | F1 | False-flag |
|---|---|---|---|---|---|
| combined | 0.52 | 0.989 | 0.947 | 0.967 | 3.4% |
| embedding-only | 0.80 | 0.785 | 0.968 | 0.867 | 86.2% |
| lexical-only | 0.39 | 1.000 | 0.957 | 0.978 | 0.0% |

Catch rate by subtype (combined): wrong_passage 1.000, hard_negative
1.000, mutated_fact 1.000, contradicted 0.286.

End-to-end: retrieval recall@3 1.000; raw unsupported rate 0.563
(27/48); unsupported catch rate 1.000; repair success 1.000 (every
repaired citation pointed at the claim's true passage); residual
unsupported rate in final answers 0.000 (0/39 surviving claims);
false-flag rate on clean claims 0.000; answer-level low-confidence
flags: precision 0.80, recall 1.00 (20/24 answers flagged); 2
abstentions where all claims were dropped.

## 4. Failure analysis (the honest part)

**Contradicted claims pass.** 5 of 7 contradicted test items were
missed. A negated claim reuses the passage's vocabulary and numbers,
so all three signals score it as supported. The scorer has no negation
handling and no entailment model; this is a real ceiling of the
approach, not a tuning issue. A production version needs an NLI or
negation-aware term, at the cost of the current determinism/simplicity.

**Lexical beats combined — on this set.** Gold claims are
extractive-style paraphrases, so token F1 is nearly a label leak. The
combined scorer's embedding term adds noise here and slightly trails
lexical-only (F1 0.967 vs 0.978). The honest reading: this benchmark
cannot demonstrate the embedding term's value; a paraphrase-heavy claim
set is needed for that, and it does not exist yet in this project.

**Small scale.** 24 passages / 24 questions. End-to-end perfection
(0 residual) is a statement about this toy regime — planted noise of
exactly the types the scorer was built for — not about LLM-generated
answers in the wild.

## 5. Limitations

- Synthetic, self-authored corpus and claims; facts are common
  knowledge, paraphrased.
- Template generator, not an LLM; fluent-but-wrong prose, multi-hop
  claims, and partial support are out of scope. "Partially supported"
  is not modelled: support is binary by construction.
- Threshold and fusion weights are fitted to this corpus family;
  transfer to other corpora is unmeasured.
- The fallback searches only the local corpus; "fall back to search"
  beyond the corpus (web) is an interface, not an implementation.

## 6. Reproduction

```bash
pip install -e .
GROUNDCHECK_MODEL=<all-MiniLM-L6-v2 dir> PYTHONPATH=src python -m groundcheck.evaluate
pytest
```

Deterministic: fixed seeds (eval set 20261026, generator 7), CPU-only,
no network calls at evaluation time. `results.json` in this repository
was produced by exactly this command.
