# Kill-gate record — groundcheck (P2)

Checked 2026-10-26 before building. Gate rule: stop if an existing
popular project — or one of habib-analyst's own repos — already does
citation-support verification with measured precision/recall **in this
form**: (1) a runtime stage inside a RAG agent loop that drives repair /
fallback / abstention, (2) scored by embedding similarity + lexical
overlap, (3) measured as citation precision/recall on an evaluation set
with known ground truth.

## Verdict: PROCEED — no exact prior art in this form. Material overlap exists and is disclosed below.

## External prior art

- **ALCE** (Princeton NLP, popular): benchmark for LLM citation
  generation; citation precision/recall computed with an 11B NLI model
  (TRUE) over LLM outputs on ASQA/QAMPARI/ELI5. It is an offline
  benchmark of *other systems' outputs*, not a runtime verifier inside
  an agent, uses NLI rather than embedding+lexical scoring, and its
  labels come from model/human judgement, not planted ground truth.
  Different form on all three axes.
- **LitRAG** (PharmaTools.AI, small/new): medical-literature RAG with a
  citation-faithfulness eval; verification = verbatim-quote existence +
  an **LLM judge**. Paid-API pattern, not a local embedding scorer, no
  repair loop.
- **ContextTrace** (small): offline verifier for pre-recorded RAG
  traces; deterministic **lexical** scoring with a small bundled
  benchmark reporting per-label P/R/F1. Explicitly *not* embeddings and
  *not* an LLM judge; it audits traces after the fact and does not sit
  inside an agent loop repairing answers.
- GitHub search (citation verification / RAG faithfulness): largest
  hits are tangential (reference-existence checkers, academic-paper
  tooling); dedicated RAG citation verifiers are all ≤ ~30 stars and
  none combines an in-loop repair pipeline with an embedding+lexical
  support scorer measured on planted ground truth.

## habib-analyst's own repos (overlap disclosed)

- **cite-check** (his): offline auditor for finished RAG answers;
  (claim, cited passage) support verdicts from key-phrase coverage +
  TF-weighted lexical overlap + negation/numeric/antonym rules,
  optional NLI backend. Overlap: the claim–citation support task
  itself. Differences: offline audit vs in-loop verification driving
  repair/abstention; lexical+rules vs embedding+lexical fusion; demo
  fixture vs a measured precision/recall benchmark on planted ground
  truth (cite-check publishes no such benchmark). Note: cite-check's
  negation handling covers exactly the class groundcheck measures
  itself failing on (contradicted claims, 28.6% caught) — the two are
  complementary, and PAPER.md says so.
- **verirag** (his): full RAG with per-claim grounding ledger scored
  against the *best-matching* evidence, calibrated refusal, measured
  refusal precision/recall on 60 golden cases. Differences: verirag
  scores claims against best evidence and refuses whole answers;
  groundcheck verifies the *specific citation attached* to each claim,
  repairs citations, and measures citation-level precision/recall on
  planted labels. Adjacent, not identical.
- **citadel-rag** (his): detects disagreement *between* sources. Not
  claim–citation support verification.

## Residual risk (for the parent to weigh)

The portfolio now holds three citation/grounding-adjacent projects
(cite-check, verirag, groundcheck). The build proceeded because the
gate's exact-form test is not met by any of them, but positioning
three such projects is a portfolio decision above this build's pay
grade; the differentiation above is stated in README.md and PAPER.md
so the distinction is public and honest.
