"""Small load test: N concurrent "clinicians" each sending clips to /transcribe.

    python scripts/load_test.py --url http://127.0.0.1:8001 --users 30 --requests 2

Prints p50 / p95 / max latency, throughput and error count. No extra tools needed.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

import httpx
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


async def user(client, url, clips, n, lat, errors):
    for i in range(n):
        f = clips[i % len(clips)]
        t0 = time.perf_counter()
        try:
            r = await client.post(f"{url}/transcribe", files={"audio": (f.name, f.read_bytes(), "audio/wav")},
                                  data={"client": "loadtest"})
            if r.status_code != 200:
                errors.append(r.status_code)
            else:
                lat.append(time.perf_counter() - t0)
        except Exception as e:
            errors.append(type(e).__name__)


async def main_async(a):
    ev = pd.read_csv(ROOT / "data" / "eval.csv", dtype={"clip_id": str})
    clips = [ROOT / "data" / "clips" / f"{c}.wav" for c in ev.clip_id]
    lat, errors = [], []
    t0 = time.perf_counter()
    async with httpx.AsyncClient(timeout=a.timeout) as client:
        await asyncio.gather(*[user(client, a.url, clips[u % len(clips):] + clips[:u % len(clips)],
                                    a.requests, lat, errors) for u in range(a.users)])
    wall = time.perf_counter() - t0
    lat.sort()
    res = {"users": a.users, "requests": a.users * a.requests, "ok": len(lat), "errors": len(errors),
           "error_kinds": sorted({str(e) for e in errors}),
           "p50_s": round(statistics.median(lat), 2) if lat else None,
           "p95_s": round(lat[int(0.95 * (len(lat) - 1))], 2) if lat else None,
           "max_s": round(lat[-1], 2) if lat else None, "throughput_rps": round(len(lat) / wall, 2),
           "wall_s": round(wall, 1)}
    print(json.dumps(res, indent=2))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=2), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8001")
    ap.add_argument("--users", type=int, default=30)
    ap.add_argument("--requests", type=int, default=2, help="requests per user")
    ap.add_argument("--timeout", type=float, default=30.0, help="client timeout (s): a clinician gives up")
    ap.add_argument("--out")
    asyncio.run(main_async(ap.parse_args()))
