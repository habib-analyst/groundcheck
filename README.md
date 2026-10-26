# groundcheck

**A RAG agent that verifies its own citations.**

Most RAG systems attach a citation and move on. But a citation is a
*claim about evidence*: it says "this passage supports this sentence."
That claim is often false — the passage is about something else, it is
the wrong passage from the same topic, or the sentence's numbers were
never in it. groundcheck treats every citation as unverified until a
local support scorer proves otherwise, flags what fails, and repairs or
drops it before the answer ships.

No LLM judge. No API calls. Everything runs locally on CPU with
all-MiniLM-L6-v2 embeddings plus lexical signals, and the whole system
is measured on a planted ground-truth evaluation set where every label
is known by construction — so the precision/recall numbers below are
exact, not judged.

## The verification idea

For each (claim, cited passage) pair, three signals are fused:

| Signal | What it catches |
|---|---|
| Embedding similarity (MiniLM cosine) | wrong-topic citations, paraphrase-aware |
| Token F1 | claims whose words are not grounded in the passage |
| Number agreement | invented figures — every number in the claim must appear in the cited passage, otherwise the score is capped |

A citation scoring below threshold triggers the **fallback path**: a
corpus-wide re-search for a passage that actually supports the claim.
Found → the citation is repaired. Not found → the claim is dropped and
the answer is flagged low-confidence. No surviving claims → the agent
abstains.

## Pipeline

```
question → retrieve (hybrid, top-3) → generate answer with citations
        → verify every citation → flag failures
        → fallback search: repair citation | drop claim
        → answer + confidence (or abstention)
```

## Measured results

Evaluation set: 24 passages, 204 planted (claim, citation) pairs —
supported, wrong-passage, same-topic hard negatives, number-mutated
facts, and contradicted claims. Threshold tuned on a 81-item dev split;
all numbers below are the 123-item test split.

**Verifier (flagging unsupported citations):**

| Scorer | Precision | Recall (catch rate) | F1 | False-flag rate |
|---|---|---|---|---|
| Combined (embedding + lexical + numbers) | 0.989 | 0.947 | 0.967 | 3.4% |
| Embedding only | 0.785 | 0.968 | 0.867 | 86.2% |
| Lexical only | 1.000 | 0.957 | 0.978 | 0.0% |

**Catch rate by failure type (combined):** wrong passage 100%,
same-topic hard negative 100%, mutated fact 100%, **contradicted
claims only 28.6%** — see Limitations; the scorer has no negation or
entailment signal, and contradicted claims share almost all words and
numbers with their passage.

**End-to-end** (24 questions, generator with planted citation noise —
30% wrong citations, 15% mutated facts):

| Metric | Value |
|---|---|
| Raw unsupported-claim rate in generated answers | 56.3% |
| Unsupported claims caught (flagged / repaired / dropped) | 100% |
| Citation repair accuracy | 100% |
| Residual unsupported rate in final answers | 0.0% |
| Clean claims falsely flagged | 0.0% |
| Answer-level low-confidence flags | precision 0.80, recall 1.00 |
| Retrieval recall@3 | 1.00 |
| Abstentions (all claims dropped) | 2 / 24 |

Full numbers in `results.json`; method and honest failure analysis in
`PAPER.md`.

## Quickstart

```bash
pip install -e .
# point at a local MiniLM model dir, or omit to download from HF:
GROUNDCHECK_MODEL=/path/to/all-MiniLM-L6-v2 PYTHONPATH=src \
  python -m groundcheck.evaluate    # writes results.json
pytest                              # 17 tests
```

## Honest limitations

- **Contradicted claims are the blind spot** (28.6% caught). A claim
  that negates its passage keeps the same words and numbers, so
  similarity-based scoring passes it. Catching these needs a negation
  or NLI signal, which this scorer deliberately does not include.
- The evaluation corpus is small and its gold claims are
  extractive-style, which flatters the lexical signal (lexical-only
  beats combined here). On heavily paraphrased claims the embedding
  term should matter more; that regime is not measured here.
- The generator is a controlled simulator with planted noise, not an
  LLM — by design, so pipeline metrics are exact. LLM-generated
  answers would add failure modes this set does not cover.
- Embedding-only verification is unusable alone (86% false-flag rate
  at its tuned threshold): semantic similarity is not support.

## Relation to prior work

ALCE (Princeton) benchmarks citation quality of LLM outputs using an
NLI model; groundcheck is instead a runtime component inside the agent
loop — verification drives repair and abstention, uses no NLI or LLM
judge, and is measured on planted ground truth rather than model
outputs. Offline citation auditors check finished answers; groundcheck
checks its own answer before returning it and fixes what fails.

## License

MIT
