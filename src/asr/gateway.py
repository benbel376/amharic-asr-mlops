"""Canary gateway: sends a share of traffic to the candidate model, the rest to the champion.

    uvicorn asr.gateway:app --port 8000

Environment:
  CHAMPION_URL      default http://127.0.0.1:8001
  CANDIDATE_URL     default http://127.0.0.1:8002
  CANARY_PERCENT    0-100, default 0 (all traffic to the champion). Set to 0 to roll back.
  SHADOW            1 = also send every request to the candidate but return the champion's answer

The client (e.g. the OpenMRS assistant app) only knows the gateway URL. Moving to a real
server changes that one URL, nothing else.
"""
from __future__ import annotations

import hashlib
import os

import httpx
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse

CHAMPION = os.environ.get("CHAMPION_URL", "http://127.0.0.1:8001")
CANDIDATE = os.environ.get("CANDIDATE_URL", "http://127.0.0.1:8002")
STATE = {"canary_percent": float(os.environ.get("CANARY_PERCENT", "0")),
         "shadow": os.environ.get("SHADOW", "0") == "1"}

app = FastAPI(title="ASR gateway")
_client = httpx.AsyncClient(timeout=120)


@app.get("/health")
async def health():
    out = {"gateway": "ok", **STATE}
    for name, url in (("champion", CHAMPION), ("candidate", CANDIDATE)):
        try:
            out[name] = (await _client.get(f"{url}/health", timeout=3)).json()
        except Exception as e:
            out[name] = {"status": f"unreachable ({type(e).__name__})"}
    return out


@app.post("/admin/canary")
async def set_canary(percent: float = Form(...), shadow: bool = Form(False)):
    """Change the split at runtime. In real life: behind authentication and a Gate 2 record."""
    STATE["canary_percent"], STATE["shadow"] = max(0.0, min(100.0, percent)), shadow
    return STATE


@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...), client: str = Form("unknown")):
    data = await audio.read()
    files = {"audio": (audio.filename or "audio.wav", data, audio.content_type or "audio/wav")}
    # Deterministic split: the same recording always goes to the same model, so a canary run can be
    # repeated exactly. In production hash the facility or clinician id instead (each user sees one version).
    bucket = int(hashlib.sha256(data).hexdigest()[:8], 16) % 100
    target = CANDIDATE if bucket < STATE["canary_percent"] else CHAMPION
    r = await _client.post(f"{target}/transcribe", files=files, data={"client": client})
    if STATE["shadow"] and target == CHAMPION:
        try:  # result is logged by the candidate service, never shown to the clinician
            await _client.post(f"{CANDIDATE}/transcribe", files=files, data={"client": "shadow"})
        except Exception:
            pass
    return JSONResponse(r.json(), status_code=r.status_code)
