"""Amharic text normaliser for scoring ASR output.

Versioned: every evaluation and every API response records NORMALIZER_VERSION,
so two WER numbers are only compared when they were scored the same way (F12).

raw / am-norm-v0: collapse whitespace only.
am-norm-v1: NFC, lower-case Latin, drop <tags>, Ge'ez numerals to Arabic digits,
    fold homophone letter families (ሀ/ሐ/ኀ/ኸ, ሰ/ሠ, አ/ዐ, ጸ/ፀ, and ሃ->ሀ, ኣ->አ),
    strip Ethiopic and ASCII punctuation.
"""
from __future__ import annotations

import re
import unicodedata

NORMALIZER_VERSION = "am-norm-v1"

# Each Ge'ez family has 7 basic orders in consecutive code points.
_FAMILY_FOLDS = [
    (0x1210, 0x1200),  # ሐ -> ሀ
    (0x1280, 0x1200),  # ኀ -> ሀ
    (0x12B8, 0x1200),  # ኸ -> ሀ
    (0x1220, 0x1230),  # ሠ -> ሰ
    (0x12D0, 0x12A0),  # ዐ -> አ
    (0x1340, 0x1338),  # ፀ -> ጸ
]
_FOLD: dict[str, str] = {}
for _src, _dst in _FAMILY_FOLDS:
    for _order in range(7):
        _FOLD[chr(_src + _order)] = chr(_dst + _order)
# 4th order (-a) of the h and glottal families sounds like the 1st order.
_FOLD["ሃ"] = "ሀ"  # ሃ -> ሀ
_FOLD["ኣ"] = "አ"  # ኣ -> አ
_FOLD["ዓ"] = "አ"  # ዓ -> አ (after ዐ folding it would be ኣ)
_FOLD_TABLE = str.maketrans(_FOLD)

# Ethiopic punctuation ፡ ። ፣ ፤ ፥ ፦ ፧ ፨ plus ASCII and typographic punctuation.
_PUNCT = re.compile("[፡-፨!-/:-@\\[-`{-~‘-‟«»]")
_TAGS = re.compile(r"<[^>]*>")
_SPACE = re.compile(r"\s+")

# Ge'ez numerals: ፩-፱ (1-9), ፲-፺ (10-90), ፻ (100), ፼ (10 000).
_GEEZ_UNITS = {chr(0x1369 + i): i + 1 for i in range(9)}
_GEEZ_TENS = {chr(0x1372 + i): (i + 1) * 10 for i in range(9)}
_GEEZ_NUM = re.compile("[፩-፼]+")


def geez_to_int(s: str) -> int:
    """Convert a Ge'ez numeral string, e.g. '፲፪' -> 12, '፳፻፲፰' -> 2018."""
    total, group = 0, 0
    for ch in s:
        if ch in _GEEZ_UNITS:
            group += _GEEZ_UNITS[ch]
        elif ch in _GEEZ_TENS:
            group += _GEEZ_TENS[ch]
        elif ch == "፻":  # ፻ hundred
            total += (group or 1) * 100
            group = 0
        elif ch == "፼":  # ፼ ten thousand
            total = (total + group or 1) * 10000
            group = 0
        else:
            raise ValueError(f"not a Ge'ez numeral: {ch!r}")
    return total + group


def normalize(text: str, version: str = NORMALIZER_VERSION) -> str:
    if version in ("raw", "am-norm-v0"):
        return _SPACE.sub(" ", text).strip()
    if version != "am-norm-v1":
        raise ValueError(f"unknown normaliser version {version}")
    t = unicodedata.normalize("NFC", text)
    t = _TAGS.sub(" ", t)
    t = _GEEZ_NUM.sub(lambda m: f" {geez_to_int(m.group())} ", t)
    t = t.translate(_FOLD_TABLE)
    t = _PUNCT.sub(" ", t)
    t = t.lower()
    return _SPACE.sub(" ", t).strip()
