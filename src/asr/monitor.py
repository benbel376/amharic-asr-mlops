"""Label-free monitoring from the request logs (and clinician edits, if any).

    python -m asr.monitor                       # print the report for logs/
    python -m asr.monitor --html results/monitor.html
    uvicorn asr.monitor:app --port 8050         # the same report as a web page (in Docker Compose)

Reads only the structured request log (no audio, no text). Per model version and per
window of N requests it shows: volume, rejected share, empty-output share, p50/p95 latency,
median signal-to-noise ratio (a microphone / channel fingerprint), and the clinician
edit rate from logs/feedback.jsonl. Each metric is checked against configs/params.yaml.
Load-test and shadow requests are left out. scripts/start_server.sh renames old logs at each start.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
P = yaml.safe_load((ROOT / "configs" / "params.yaml").read_text(encoding="utf-8"))["monitoring"]
LOG_DIR = Path(os.environ.get("ASR_LOG_DIR", str(ROOT / "logs")))


def _read(name: str) -> pd.DataFrame:
    frames = []
    for f in sorted(LOG_DIR.glob(f"**/{name}")):
        rows = [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines() if x.strip()]
        frames.append(pd.DataFrame(rows))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def report(window: int = 40) -> pd.DataFrame:
    req = _read("requests.jsonl")
    if req.empty:
        return pd.DataFrame()
    if "client" in req:  # shadow copies and load tests are not clinic traffic
        req = req[~req.client.isin(["shadow", "loadtest"])]
    fb = _read("feedback.jsonl")
    rows = []
    for ver, g in req.groupby("model_version", sort=False):
        g = g.reset_index(drop=True)
        for start in range(0, len(g), window):
            w = g.iloc[start:start + window]
            ok = w[w.status == "ok"]
            row = {"model_version": ver, "window": f"{start + 1}-{start + len(w)}", "requests": len(w),
                   "rejected": round((w.status == "rejected").mean(), 3),
                   "empty_rate": round(ok["empty"].mean(), 3) if len(ok) else None,
                   "p50_ms": ok.latency_ms.quantile(0.5) if len(ok) else None,
                   "p95_ms": ok.latency_ms.quantile(0.95) if len(ok) else None,
                   "snr_db_median": round(w.snr_db.median(), 1) if "snr_db" in w else None}
            if not fb.empty:
                e = fb[fb.request_id.isin(w.request_id)]
                row["edit_rate"] = round(e.edit_rate.mean(), 3) if len(e) else None
            alerts = []
            if row["empty_rate"] is not None and row["empty_rate"] > P["max_empty_rate"]:
                alerts.append("EMPTY")
            if row["snr_db_median"] is not None and row["snr_db_median"] < P["min_snr_db_median"]:
                alerts.append("CHANNEL")
            if row.get("edit_rate") is not None and row["edit_rate"] > P["max_edit_rate"]:
                alerts.append("EDITS")
            row["alerts"] = ",".join(alerts) or "-"
            rows.append(row)
    return pd.DataFrame(rows)


def to_html(df: pd.DataFrame) -> str:
    table = df.to_html(index=False, na_rep="-") if not df.empty else "<p>No requests logged yet.</p>"
    return ("<!doctype html><meta charset='utf-8'><title>ASR monitor</title>"
            "<style>body{font-family:Segoe UI,Roboto,sans-serif;background:#F6F6F6;color:#1C1C1C;margin:24px}"
            "table{border-collapse:collapse;background:#fff}td,th{padding:6px 10px;border-bottom:1px solid #ddd}"
            "th{background:#1C1C1C;color:#fff}</style><h2>Amharic ASR - label-free monitor</h2>"
            f"<p>Thresholds: empty &gt; {P['max_empty_rate']}, median SNR &lt; {P['min_snr_db_median']} dB, "
            f"edit rate &gt; {P['max_edit_rate']}. Refresh to update.</p>{table}")


try:
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse

    app = FastAPI(title="ASR monitor")

    @app.get("/", response_class=HTMLResponse)
    def index():
        return to_html(report())
except ImportError:  # pragma: no cover
    app = None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=40)
    ap.add_argument("--html")
    a = ap.parse_args()
    df = report(a.window)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(df.to_string(index=False) if not df.empty else "no requests logged yet")
    if a.html:
        Path(a.html).write_text(to_html(df), encoding="utf-8")
        print(f"wrote {a.html}")


if __name__ == "__main__":
    main()
