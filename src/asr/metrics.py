"""WER / CER with an explicit normaliser version, plus edit rate and term recall."""
from __future__ import annotations

import jiwer

from .normalize import normalize


def _prep(refs, hyps, version):
    pairs = [(normalize(a, version), normalize(b, version)) for a, b in zip(refs, hyps)]
    pairs = [(a, b) for a, b in pairs if a]  # jiwer needs a non-empty reference
    return [a for a, _ in pairs], [b for _, b in pairs]


def wer(refs, hyps, version="am-norm-v1") -> float:
    r, h = _prep(refs, hyps, version)
    return float(jiwer.wer(r, h))


def cer(refs, hyps, version="am-norm-v1") -> float:
    r, h = _prep(refs, hyps, version)
    return float(jiwer.cer(r, h))


def edit_rate(original: str, corrected: str, version="am-norm-v1") -> float:
    """Share of characters the clinician changed: CER of the model output vs the corrected text."""
    a, b = normalize(corrected, version), normalize(original, version)
    if not a:
        return 0.0 if not b else 1.0
    return float(jiwer.cer(a, b))


def term_recall(terms_per_utt, hyps, version="am-norm-v1") -> float:
    """Share of expected terms that appear (after normalisation) in the hypothesis."""
    found = total = 0
    for terms, h in zip(terms_per_utt, hyps):
        nh = normalize(h, version)
        for t in terms:
            total += 1
            found += normalize(t, version) in nh
    return found / total if total else float("nan")
