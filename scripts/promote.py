"""Compare candidate vs champion with the written promotion rule (configs/params.yaml).

    python scripts/promote.py                     # results/tiny-600hr.json vs results/tiny-stage2.json
    python scripts/promote.py --candidate tiny-600hr-int8

Prints PROMOTE or REJECT with reasons and writes results/promotion_<candidate>.json.
The script never changes what is deployed: a named person signs docs/approvals/ (Gate 2).
Exit code 1 on REJECT, so CI can block a merge.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
P = yaml.safe_load((ROOT / "configs" / "params.yaml").read_text(encoding="utf-8"))


def load(name: str) -> dict:
    for folder in ("results", "results/precomputed"):
        f = ROOT / folder / f"{name}.json"
        if f.exists():
            return json.loads(f.read_text(encoding="utf-8"))
    raise SystemExit(f"no results for {name}: run python -m asr.evaluate first (or use results/precomputed)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--champion", default=P["models"]["champion"]["id"])
    ap.add_argument("--candidate", default=P["models"]["candidate"]["id"])
    a = ap.parse_args()
    ch, ca, rule = load(a.champion), load(a.candidate), P["promotion"]
    reasons = []
    if ch["normalizer"] != ca["normalizer"]:
        reasons.append(f"different normalisers ({ch['normalizer']} vs {ca['normalizer']}): not comparable")
    for k, lim in (("wer", rule["max_wer_increase"]), ("cer", rule["max_cer_increase"])):
        d = ca["overall"][k] - ch["overall"][k]
        print(f"{k.upper():4s} champion {ch['overall'][k]:.3f}  candidate {ca['overall'][k]:.3f}  diff {d:+.3f}")
        if d > lim:
            reasons.append(f"overall {k.upper()} worse by {d:+.3f}")
    for g in sorted(ch["groups"]):
        d = ca["groups"][g]["cer"] - ch["groups"][g]["cer"]
        flag = "  <-- worse" if d > rule["max_group_cer_increase"] else ""
        print(f"  {g:14s} CER champion {ch['groups'][g]['cer']:.3f}  candidate {ca['groups'][g]['cer']:.3f}"
              f"  diff {d:+.3f}{flag}")
        if flag:
            reasons.append(f"group {g} CER worse by {d:+.3f}")
    decision = "REJECT" if reasons else "PROMOTE"
    print(f"\nDecision: {decision}" + "".join(f"\n  - {r}" for r in reasons))
    out = {"champion": a.champion, "candidate": a.candidate, "decision": decision, "reasons": reasons,
           "rule": rule}
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / f"promotion_{a.candidate}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return 0 if decision == "PROMOTE" else 1


if __name__ == "__main__":
    sys.exit(main())
