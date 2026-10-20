"""Data gate for the evaluation set (DATA-03): run before every evaluation.

    python scripts/check_data.py
    python scripts/check_data.py --samples     # the gate's verdict on each file in data/gate_samples/

Checks: manifest columns and row count, every eval clip exists and passes the audio gate,
no empty references, no duplicate sentences in the eval set, references contain Ge'ez text,
and the eval set's median SNR is in the expected range. Exit code 1 on failure.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from asr.audio import gate  # noqa: E402

GEEZ = re.compile("[ሀ-፿]")


def samples() -> int:
    for f in sorted((ROOT / "data" / "gate_samples").glob("*")):
        g, _ = gate(str(f))
        verdict = "REJECT" if not g.ok else ("WARN  " if g.warnings else "PASS  ")
        print(f"{f.name:16s} {verdict} {'; '.join(g.issues + g.warnings)}")
    return 0


def main() -> int:
    if "--samples" in sys.argv:
        return samples()
    fails = []
    m = pd.read_csv(ROOT / "data" / "manifest.csv", dtype={"clip_id": str})
    ev = pd.read_csv(ROOT / "data" / "eval.csv", dtype={"clip_id": str})
    need = {"clip_id", "sentence_id", "text", "gender", "duration_s"}
    if not need <= set(ev.columns):
        fails.append(f"missing columns {need - set(ev.columns)}")
    if len(m) != 223:
        fails.append(f"manifest has {len(m)} rows, expected 223 (FLEURS am dev)")
    if ev.text.isna().any() or (ev.text.str.strip() == "").any():
        fails.append("empty reference transcripts")
    if ev.sentence_id.duplicated().any():
        fails.append("duplicate sentences in the eval set")
    if not ev.text.map(lambda t: bool(GEEZ.search(str(t)))).all():
        fails.append("reference without Ge'ez characters")
    snr = []
    for cid in ev.clip_id:
        f = ROOT / "data" / "clips" / f"{cid}.wav"
        if not f.exists():
            fails.append(f"missing audio {cid}")
            continue
        g, _ = gate(str(f))
        snr.append(g.features.get("snr_db", 0))
        if not g.ok:
            fails.append(f"{cid}: {'; '.join(g.issues)}")
    med = float(np.median(snr)) if snr else 0
    if not 25 <= med <= 60:
        fails.append(f"median SNR {med:.1f} dB outside 25-60 dB: wrong audio in the eval folder?")
    print(f"eval clips: {len(ev)}  median SNR {med:.1f} dB  ->  {'PASS' if not fails else 'FAIL'}")
    for f in fails[:20]:
        print("  -", f)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
