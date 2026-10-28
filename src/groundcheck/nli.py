"""Local NLI stage — the v2 addition.

v1's scorer fuses embedding similarity, token F1 and number agreement.
None of those signals can see negation: a claim that *contradicts* its
passage reuses the passage's words and numbers, so it sails through
(v1 catch rate on contradicted claims: 0.286). This module wraps a
small pretrained NLI model that runs fully locally on CPU and returns
P(contradiction) for a (passage, claim) pair, used by the verifier as
a veto on top of the v1 score.

Default model: cross-encoder/nli-MiniLM2-L6-H768 — a MiniLMv2-L6
encoder (distilled from RoBERTa-Large) fine-tuned for 3-way NLI
(contradiction / entailment / neutral), ~82M parameters. Point
GROUNDCHECK_NLI_MODEL at a local model directory (recommended) or an
HF model id. Label indices are read from the model's own config —
never hard-coded — so another MNLI-style checkpoint can be dropped in.

No API calls. Inference is deterministic (eval mode, no sampling).
"""

import os

DEFAULT_NLI_MODEL = os.environ.get(
    "GROUNDCHECK_NLI_MODEL", "cross-encoder/nli-MiniLM2-L6-H768"
)

_MAX_LEN = 256
_BATCH = 16


class NLIClassifier:
    def __init__(self, model_name=DEFAULT_NLI_MODEL):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self.model_name = model_name
        self._tok = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self._model.eval()
        id2label = self._model.config.id2label
        self._idx = {
            label.lower(): idx for idx, label in id2label.items()
        }
        missing = {"contradiction", "entailment", "neutral"} - set(self._idx)
        if missing:
            raise ValueError(
                f"NLI model {model_name!r} is missing labels {sorted(missing)}; "
                f"config id2label is {id2label}"
            )

    # -- inference ------------------------------------------------------
    def probs_batch(self, pairs):
        """pairs: list of (premise, hypothesis). Returns a list of
        dicts with contradiction / entailment / neutral probabilities."""
        torch = self._torch
        out = []
        for i in range(0, len(pairs), _BATCH):
            chunk = pairs[i : i + _BATCH]
            enc = self._tok(
                [p for p, _ in chunk],
                [h for _, h in chunk],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=_MAX_LEN,
            )
            with torch.no_grad():
                logits = self._model(**enc).logits
            probs = torch.softmax(logits, dim=-1).tolist()
            for row in probs:
                out.append(
                    {label: float(row[idx]) for label, idx in self._idx.items()}
                )
        return out

    def probs(self, premise, hypothesis):
        return self.probs_batch([(premise, hypothesis)])[0]

    def contradiction_probs(self, pairs):
        """Just P(contradiction) for each (premise, hypothesis) pair."""
        return [p["contradiction"] for p in self.probs_batch(pairs)]
