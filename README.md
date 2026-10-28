# groundcheck

**A RAG agent that verifies its own citations.**

Most RAG systems attach a citation and move on. But a citation is a
*claim about evidence*: it says "this passage supports this sentence."
That claim is often false — the passage is about something else, it is
the wrong passage from the same topic, or the sentence's numbers were
never in it. groundcheck treats every citation as unverified until a
local support scorer proves otherwise, flags what fails, and repairs or
drops it before the answer ships.

No LLM judge. No API calls. Everything runs locally on CPU, and the
whole system is measured on a planted ground-truth evaluation set where
every label is known by construction — so the precision/recall numbers
below are exact, not judged.

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

### v2: the NLI contradiction veto

v1's blind spot was **contradicted claims** (28.6% caught): a claim
that negates its passage keeps the same words and numbers, so
similarity scores it as supported. v2 adds a local NLI stage
(`cross-encoder/nli-MiniLM2-L6-H768`, premise = passage, hypothesis =
claim) whose P(contradiction) acts as a **veto**:

```
flag as unsupported  iff  v1_score < tau  OR  P_NLI(contradiction) >= tau_nli
```

The veto is not blended into the score — a weighted sum would let high
similarity outvote a contradiction, which is exactly the v1 failure.
The same veto gates fallback repairs: a claim may not be repaired onto
a passage that contradicts it. Bars and the tuning protocol were
pre-registered in `PREREG.md` before the v2 run.

## Pipeline

```
question → retrieve (hybrid, top-3) → generate answer with citations
        → verify every citation (v1 score + NLI veto) → flag failures
        → fallback search: repair citation | drop claim
        → answer + confidence (or abstention)
```

## Measured results

Evaluation set: 24 passages, 204 planted (claim, citation) pairs —
supported, wrong-passage, same-topic hard negatives, number-mutated
facts, and contradicted claims. Thresholds tuned on an 81-item dev
split (tau = 0.52 as in v1; tau_nli = 0.65 by the PREREG protocol);
all numbers below are the 123-item test split. v1 numbers are
recomputed by the same run and reproduce v1 exactly.

**Verifier (flagging unsupported citations):**

| Scorer | Precision | Recall (catch rate) | F1 | False-flag rate |
|---|---|---|---|---|
| **v2 (v1 combined + NLI veto)** | 0.979 | **1.000** | **0.990** | 6.9% |
| v1 combined (embedding + lexical + numbers) | 0.989 | 0.947 | 0.967 | 3.4% |
| Embedding only | 0.785 | 0.968 | 0.867 | 86.2% |
| Lexical only | 1.000 | 0.957 | 0.978 | 0.0% |

**Catch rate by failure type:** wrong passage 100%, same-topic hard
negative 100%, mutated fact 100% (all unchanged from v1), and
**contradicted claims 28.6% → 100%** (7/7 test items — a small
denominator; see Limitations).

**End-to-end** (24 questions, generator with planted citation noise —
30% wrong citations, 15% mutated facts):

| Metric | v1 | v2 |
|---|---|---|
| Raw unsupported-claim rate in generated answers | 56.3% | 56.3% |
| Unsupported claims caught (flagged / repaired / dropped) | 100% | 100% |
| Citation repair accuracy | 100% | 100% |
| Residual unsupported rate in final answers | 0.0% | 0.0% |
| Clean claims falsely flagged | 0.0% | 4.8% (1/21) |
| Answer-level low-confidence flags | P 0.80, R 1.00 | P 0.76, R 1.00 |
| Retrieval recall@3 | 1.00 | 1.00 |
| Abstentions (all claims dropped) | 2 / 24 | 2 / 24 |

Full numbers in `results.json`; method, item-level failure analysis
and the locked bars in `PAPER.md` and `PREREG.md`.

## Quickstart

```bash
pip install -e .
# point at a local MiniLM model dir, or omit to download from HF:
GROUNDCHECK_MODEL=/path/to/all-MiniLM-L6-v2 \
PYTHONPATH=src python -m groundcheck.evaluate    # v1-only eval

# v2 also needs the local NLI model (~330 MB, fetched once):
mkdir -p models/nli-MiniLM2-L6-H768
for f in config.json model.safetensors tokenizer.json \
         tokenizer_config.json special_tokens_map.json vocab.json merges.txt; do
  curl -sL "https://huggingface.co/cross-encoder/nli-MiniLM2-L6-H768/resolve/main/$f" \
       -o "models/nli-MiniLM2-L6-H768/$f"
done
GROUNDCHECK_MODEL=/path/to/all-MiniLM-L6-v2 \
GROUNDCHECK_NLI_MODEL=$PWD/models/nli-MiniLM2-L6-H768 \
  PYTHONPATH=src python -m groundcheck.evaluate  # writes results.json
pytest                              # 22 tests
```

## Honest limitations

- **The contradiction fix is measured on 7 test items.** 100% means
  7/7 on a small, hand-written set; the honest confidence interval is
  wide. The veto and v1 cover *different* items: the NLI model itself
  misses an attributed falsehood ("information can travel faster than
  light, according to relativity" — P(contradiction) = 0.037) that the
  v1 score happens to catch, and it misses vague-quantifier
  contradictions ("a negligible share") entirely.
- **The veto false-fires, rarely and specifically.** Its one test-split
  false positive is a true claim ("Tim Berners-Lee invented the World
  Wide Web in 1989 at CERN", P(contradiction) = 0.974 against the
  passage stating exactly that). That single error is the entire
  precision dip (0.989 → 0.979) and the one falsely flagged clean claim
  end-to-end. On a larger corpus this rate is what would need watching.
- The evaluation corpus is small and its gold claims are
  extractive-style, which flatters the lexical signal (lexical-only
  beats v1-combined here). On heavily paraphrased claims the embedding
  term should matter more; that regime is not measured here.
- The generator is a controlled simulator with planted noise, not an
  LLM — by design, so pipeline metrics are exact. LLM-generated
  answers would add failure modes this set does not cover.
- v2 costs more than v1: a ~330 MB NLI model and a forward pass per
  checked citation. v1's embeddings-only simplicity was the price of
  its blind spot.
- Embedding-only verification is unusable alone (86% false-flag rate
  at its tuned threshold): semantic similarity is not support.

## Relation to prior work

ALCE (Princeton) benchmarks citation quality of LLM outputs using an
NLI model; groundcheck is instead a runtime component inside the agent
loop — verification drives repair and abstention — and is measured on
planted ground truth rather than model outputs. Offline citation
auditors check finished answers; groundcheck checks its own answer
before returning it and fixes what fails. (v1 of this project used no
NLI at all; v2 uses it as a runtime veto, not an offline judge.)

## License

MIT
