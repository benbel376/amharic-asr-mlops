"""Simulate a few days of clinic traffic against the running service, with clinician corrections.

    python scripts/simulate_clinic.py --url http://127.0.0.1:8000            # the 4-day plan in configs/params.yaml
    python scripts/simulate_clinic.py --url http://127.0.0.1:8000 --days 1   # only the first day

Day types (configs/params.yaml, clinic.plan):
  headset   the usual headsets (clean clips)
  phones    a new batch of cheap phones (the phone-channel copies), plus a few silent recordings
  outbreak  the usual headsets, but clinicians' notes now contain new terms the model never learned
            (simulated at text level: we put the terms into the saved note)

After each answer the "clinician" corrects the transcript; here the correction is the FLEURS reference.
Only the edit rate and the clip id are stored (logs/feedback.jsonl), never the text. The watcher
(src/asr/watcher.py) reads this file and decides what to do.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import httpx
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from asr.metrics import edit_rate  # noqa: E402

P = yaml.safe_load((ROOT / "configs" / "params.yaml").read_text(encoding="utf-8"))
C = P["clinic"]
LABEL = {"headset": "usual headset", "phones": "new phones", "outbreak": "outbreak notes"}


def saved_note(reference: str, day_type: str) -> str:
    """What the clinician saves. On outbreak days, every k-th word becomes a new term."""
    if day_type != "outbreak":
        return reference
    words = reference.split()
    for j, pos in enumerate(range(1, len(words), C["outbreak_every"])):
        words[pos] = C["outbreak_terms"][j % len(C["outbreak_terms"])]
    return " ".join(words)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--days", type=int, default=len(C["plan"]))
    ap.add_argument("--per-day", type=int, default=20)
    ap.add_argument("--log-dir", default=str(ROOT / "logs"))
    a = ap.parse_args()
    ev = pd.read_csv(ROOT / "data" / "eval.csv", dtype={"clip_id": str})
    fb_path = Path(a.log_dir) / "feedback.jsonl"
    fb_path.parent.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=120) as c:
        for day in range(1, a.days + 1):
            kind = C["plan"][(day - 1) % len(C["plan"])]
            folder = "shift" if kind == "phones" else "clips"
            n_ok = n_empty = n_rej = 0
            edits = []
            for i in range(a.per_day):
                row = ev.iloc[i % len(ev)]
                silent = kind == "phones" and i % 10 == 0
                path = ROOT / "data" / ("gate_samples/silence.wav" if silent else f"{folder}/{row.clip_id}.wav")
                r = c.post(f"{a.url}/transcribe", files={"audio": (path.name, path.read_bytes(), "audio/wav")},
                           data={"client": f"clinic-day{day}"})
                if r.status_code == 422:
                    n_rej += 1
                    continue
                out = r.json()
                n_ok += 1
                n_empty += not out["text"].strip()
                if silent:
                    continue  # nothing to correct
                e = edit_rate(out["text"], saved_note(row.text, kind))
                edits.append(e)
                with fb_path.open("a", encoding="utf-8") as fb:  # written at once, so the watcher sees it live
                    fb.write(json.dumps({"request_id": out["request_id"], "model_version": out["model_version"],
                                         "clip_id": row.clip_id, "edit_rate": round(e, 4), "day": day}) + "\n")
            print(f"day {day} ({LABEL[kind]}): ok {n_ok}, rejected {n_rej}, empty {n_empty}, "
                  f"mean clinician edit rate {sum(edits) / max(1, len(edits)):.3f}")


if __name__ == "__main__":
    main()
