"""Evaluate one model on the fixed eval subset; log everything to MLflow.

    python -m asr.evaluate --model champion
    python -m asr.evaluate --model candidate
    python -m asr.evaluate --model champion --quantize      # int8 version
    python -m asr.evaluate --model champion --channel clean # skip the phone-channel copy
    python -m asr.evaluate --model champion --channel clean --max-new-tokens 225   # episode 1 aside

Writes results/<run_name>.json (metrics) and results/<run_name>_hyps.csv (hypotheses).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .audio import gate
from .metrics import cer, wer
from .normalize import NORMALIZER_VERSION

ROOT = Path(__file__).resolve().parents[2]
P = yaml.safe_load((ROOT / "configs" / "params.yaml").read_text(encoding="utf-8"))
log = logging.getLogger("asr.evaluate")


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "no-git"


def run_model(engine, ev: pd.DataFrame, folder: str, bs: int, max_new_tokens: int):
    hyps, cut, secs = [], [], 0.0
    audio = []
    for cid in ev["clip_id"]:
        _, mono = gate(str(ROOT / "data" / folder / f"{cid}.wav"))
        audio.append(mono)
    for i in range(0, len(audio), bs):
        t0 = time.perf_counter()
        hyps += engine.transcribe(audio[i:i + bs], max_new_tokens=max_new_tokens)
        secs += time.perf_counter() - t0
        cut += engine.last_cut
    return hyps, cut, secs


def score(df: pd.DataFrame) -> dict:
    return {"wer": wer(df.text, df.hyp), "cer": cer(df.text, df.hyp),
            "wer_raw": wer(df.text, df.hyp, "raw"), "cer_raw": cer(df.text, df.hyp, "raw"),
            "empty_rate": float((df.hyp.str.strip() == "").mean()),
            "cut_rate": float(df.cut.mean()), "n": len(df)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="champion", choices=list(P["models"]))
    ap.add_argument("--quantize", action="store_true")
    ap.add_argument("--channel", default="both", choices=["clean", "shift", "both"])
    ap.add_argument("--n", type=int, default=None, help="use only the first n clips (smoke test)")
    ap.add_argument("--max-new-tokens", type=int, default=P["decode"]["max_new_tokens"],
                    help="decoder output limit (default from configs/params.yaml)")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")

    from .engine import WhisperEngine  # heavy import only when evaluating
    m = P["models"][a.model]
    engine = WhisperEngine(str(ROOT / m["path"]), quantize=a.quantize)
    ev = pd.read_csv(ROOT / "data" / "eval.csv", dtype={"clip_id": str})
    if a.n:
        ev = ev.head(a.n)
    ev["length"] = np.where(ev.duration_s < ev.duration_s.median(), "short", "long")

    frames, timing = [], {}
    channels = ["clean", "shift"] if a.channel == "both" else [a.channel]
    for ch in channels:
        hyps, cut, secs = run_model(engine, ev, "clips" if ch == "clean" else "shift", P["eval"]["batch_size"],
                                    a.max_new_tokens)
        d = ev.assign(hyp=hyps, cut=cut, channel="phone" if ch == "shift" else "clean")
        frames.append(d)
        timing[ch] = secs / ev.duration_s.sum()  # real-time factor
    df = pd.concat(frames)

    name = f"{m['id']}{'-int8' if a.quantize else ''}"
    if a.max_new_tokens != P["decode"]["max_new_tokens"]:
        name += f"-t{a.max_new_tokens}"  # never overwrite the configured run
    res = {"run_name": name, "model_id": m["id"], "quantized": a.quantize,
           "normalizer": NORMALIZER_VERSION, "max_new_tokens": a.max_new_tokens, "size_mb": engine.size_mb(),
           "rtf_clean": round(timing.get("clean", float("nan")), 4), "git": git_commit(),
           "overall": score(df[df.channel == "clean"]) if "clean" in channels else None,
           "groups": {}}
    for (col, val), g in df.groupby(["channel", "length"]):
        res["groups"][f"{col}/{val}"] = score(g)
    for ch, g in df.groupby("channel"):
        res["groups"][ch] = score(g)

    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    (out / f"{name}.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    df[["clip_id", "channel", "length", "text", "hyp", "cut"]].to_csv(out / f"{name}_hyps.csv", index=False,
                                                                encoding="utf-8")

    import mlflow
    mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")
    mlflow.set_experiment("amharic-asr-eval")
    with mlflow.start_run(run_name=name):
        mlflow.log_params({"model_id": m["id"], "model_path": m["path"], "quantized": a.quantize,
                           "normalizer": NORMALIZER_VERSION, "max_new_tokens": a.max_new_tokens,
                           "eval_set": "fleurs-am-dev-40",
                           "n_clips": len(ev), "git": res["git"]})
        flat = {"size_mb": res["size_mb"], "rtf_clean": res["rtf_clean"]}
        for gname, s in res["groups"].items():
            for k in ("wer", "cer", "wer_raw", "cer_raw", "empty_rate", "cut_rate"):
                flat[f"{gname.replace('/', '_')}_{k}"] = s[k]
        mlflow.log_metrics(flat)
        mlflow.log_artifact(str(out / f"{name}.json"))
    o = res["overall"] or res["groups"]["phone"]
    print(f"{name}: WER {o['wer']:.3f} (raw {o['wer_raw']:.3f})  CER {o['cer']:.3f} (raw {o['cer_raw']:.3f})"
          f"  size {res['size_mb']} MB  RTF {res['rtf_clean']}")
    for gname, s in sorted(res["groups"].items()):
        print(f"  {gname:14s} WER {s['wer']:.3f}  CER {s['cer']:.3f}  empty {s['empty_rate']:.2f}"
              f"  cut {round(s['cut_rate'] * s['n']):2d}/{s['n']}")


if __name__ == "__main__":
    main()
