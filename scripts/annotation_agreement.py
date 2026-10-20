"""Annotation quality: two annotators transcribe the same clips. Do they agree?

    python scripts/annotation_agreement.py

Builds a small two-annotator set from the eval references (annotator A writes the reference;
annotator B makes the kinds of differences real annotators make: homophone letters, punctuation,
digits instead of number words, a dropped word, an untagged name). Then:
  1. agreement as CER between A and B, raw vs normalised (am-norm-v1)
  2. guideline checks (docs/asr-annotation-guidelines summary in LAB_GUIDE episode 8)
  3. clips above the disagreement threshold go to adjudication.
Label Studio does the same at scale (two annotators per task, agreement, review queue).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import jiwer
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from asr.normalize import normalize  # noqa: E402

SWAPS = str.maketrans({"ሀ": "ሐ", "ሰ": "ሠ", "አ": "ዐ", "ሶ": "ሦ"})
RULES = {
    "G1 numbers as words in verbatim layer": re.compile(r"\d"),
    "G2 no untagged personal names (use <name>)": re.compile(r"\b(Abebe|አበበ)\b"),
    "G3 Latin in lower case": re.compile(r"[A-Z]"),
}


def annotator_b(text: str, i: int) -> str:
    t = text
    if i % 2 == 0:
        t = t.translate(SWAPS)                 # homophone spelling: allowed, scorer folds it
    if i % 3 == 0:
        t = t.replace("።", "").replace("፣", "")  # punctuation: allowed
    if i % 7 == 0:
        t = t.replace("ሦስት", "3").replace("ሶስት", "3").replace("አንድ", "1")  # breaks G1
        t = t + " 2"
    if i % 9 == 0:
        w = t.split()
        t = " ".join(w[:2] + w[3:])           # a dropped word: a real disagreement
    if i == 5:
        t = "አበበ " + t                         # breaks G2
    if i == 11:
        t = t + " BP"                          # breaks G3
    return t


def main() -> None:
    ev = pd.read_csv(ROOT / "data" / "eval.csv", dtype={"clip_id": str}).head(20)
    a = ev.text.tolist()
    b = [annotator_b(t, i) for i, t in enumerate(a)]
    out = ROOT / "data" / "annotation"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"clip_id": ev.clip_id, "text": a}).to_csv(out / "annotator_a.csv", index=False, encoding="utf-8")
    pd.DataFrame({"clip_id": ev.clip_id, "text": b}).to_csv(out / "annotator_b.csv", index=False, encoding="utf-8")
    raw = [jiwer.cer(x, y) for x, y in zip(a, b)]
    norm = [jiwer.cer(normalize(x), normalize(y)) for x, y in zip(a, b)]
    print(f"clips: {len(a)}")
    print(f"disagreement (CER between annotators)  raw: {sum(raw) / len(raw):.3f}   "
          f"normalised: {sum(norm) / len(norm):.3f}")
    print(f"clips that look different (raw CER > 0.02): {sum(r > 0.02 for r in raw)}   "
          f"really different (normalised CER > 0.02): {sum(n > 0.02 for n in norm)}")
    for rule, pat in RULES.items():
        hits = [ev.clip_id.iloc[i] for i, t in enumerate(b) if pat.search(t)]
        print(f"  {rule}: {len(hits)} transcript(s) by annotator B break it")
    adj = ev.assign(cer_norm=norm)[lambda d: d.cer_norm > 0.02][["clip_id", "cer_norm"]]
    adj.to_csv(out / "adjudication.csv", index=False)
    print(f"-> {len(adj)} clips to adjudication: data/annotation/adjudication.csv")


if __name__ == "__main__":
    main()
