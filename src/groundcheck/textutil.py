"""Text utilities: tokenisation, lexical overlap, number handling."""

import re

STOPWORDS = {
    "a", "an", "the", "of", "in", "on", "at", "to", "by", "for", "with",
    "is", "are", "was", "were", "be", "been", "it", "its", "as", "and",
    "or", "about", "per", "from", "into", "over", "under", "than",
    "each", "every", "all", "any", "this", "that", "these", "those",
    "can", "could", "may", "might", "will", "would", "has", "have",
    "had", "do", "does", "did", "not", "no", "only", "also", "roughly",
    "approximately", "around", "first", "second",
}

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")
_NUMBER_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")


def tokens(text):
    return _TOKEN_RE.findall(text.lower())


def content_tokens(text):
    """Tokens minus stopwords; numbers are kept (they carry facts)."""
    return [t for t in tokens(text) if t not in STOPWORDS]


def token_f1(claim, passage):
    """F1 between the claim's content-token set and the passage's."""
    c = set(content_tokens(claim))
    p = set(content_tokens(passage))
    if not c or not p:
        return 0.0
    inter = len(c & p)
    if inter == 0:
        return 0.0
    prec = inter / len(c)
    rec = inter / len(p)
    return 2 * prec * rec / (prec + rec)


def extract_numbers(text):
    """Numbers as floats, commas stripped ('10,935' -> 10935.0, '1970s' -> 1970.0)."""
    out = []
    for m in _NUMBER_RE.findall(text):
        try:
            out.append(float(m.replace(",", "")))
        except ValueError:
            pass
    return out


def number_agreement(claim, passage):
    """1.0 if every number in the claim appears in the passage,
    0.0 if the claim has numbers and at least one is absent,
    0.5 (neutral) if the claim has no numbers."""
    cn = extract_numbers(claim)
    if not cn:
        return 0.5
    pn = set(extract_numbers(passage))
    return 1.0 if all(any(abs(x - y) < 1e-9 for y in pn) for x in cn) else 0.0


def mutate_first_number(text, factor=1.37, alt_factor=1.53, avoid_text=None):
    """Deterministically perturb the first number in `text`.

    Used to plant fact-mutated claims. Ensures the new number does not
    appear in `avoid_text` (the supporting passage) when given.
    """
    m = _NUMBER_RE.search(text)
    if not m:
        return None
    raw = m.group(0)
    value = float(raw.replace(",", ""))
    for f in (factor, alt_factor, 2.11):
        new_val = value * f
        new_raw = _format_like(new_val, raw)
        if avoid_text is not None and new_raw.replace(",", "") in {
            n.replace(",", "") for n in _NUMBER_RE.findall(avoid_text)
        }:
            continue
        return text[: m.start()] + new_raw + text[m.end():]
    return None


def _format_like(value, raw):
    if "." in raw:
        dec = len(raw.split(".")[1].replace(",", ""))
        s = f"{value:.{dec}f}"
    else:
        s = f"{round(value):,}" if "," in raw else f"{round(value):.0f}"
    return s
