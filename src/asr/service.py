"""FastAPI transcription service: /transcribe, /health, /metrics.

    uvicorn asr.service:app --port 8001          (from src/, or with PYTHONPATH=src)

Environment:
  ASR_MODEL_PATH     model folder (default models/tiny-600hr)
  ASR_MODEL_VERSION  version label returned in every response (default tiny-600hr)
  ASR_QUANTIZE       1 = int8 dynamic quantisation
  ASR_LOG_DIR        where requests.jsonl is written (default logs/)
  ASR_LOG_TEXT       0 (default). 1 logs transcripts: the WRONG PATH in episode 7. Never in production.
  ASR_SKIP_GATE      0 (default). 1 disables the audio gate: the WRONG PATH in episode 3.
  ASR_MAX_WAITING    0 (default, unlimited). N > 0: answer 503 "busy" at once when N requests wait.

What is logged per request (LOG-01, SEC-02): request id, model version, normaliser version,
latency, audio duration, signal features (level, clipping, share of energy above 4 kHz),
gate result, output length and whether it was empty. Never the audio, never the text.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import PlainTextResponse

from .audio import gate
from .normalize import NORMALIZER_VERSION

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = os.environ.get("ASR_MODEL_PATH", str(ROOT / "models" / "tiny-600hr"))
MODEL_VERSION = os.environ.get("ASR_MODEL_VERSION", Path(MODEL_PATH).name)
QUANTIZE = os.environ.get("ASR_QUANTIZE", "0") == "1"
LOG_TEXT = os.environ.get("ASR_LOG_TEXT", "0") == "1"
SKIP_GATE = os.environ.get("ASR_SKIP_GATE", "0") == "1"
MAX_WAITING = int(os.environ.get("ASR_MAX_WAITING", "0"))  # 0 = unlimited queue (the naive default)
LOG_DIR = Path(os.environ.get("ASR_LOG_DIR", str(ROOT / "logs")))
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("asr.service")
_req_file = LOG_DIR / "requests.jsonl"
_lock = threading.Lock()
_engine_lock = threading.Lock()   # one inference at a time per process (CPU-bound)
_engine = None
_counters = {"requests_total": 0, "rejected_total": 0, "empty_total": 0, "errors_total": 0, "busy_total": 0}
_state = {"waiting": 0}
_latencies: list[float] = []



_load_lock = threading.Lock()     # the preload thread and an early request must not load the model twice


def engine():
    global _engine
    if _engine is None:
        with _load_lock:
            if _engine is None:
                from .engine import WhisperEngine
                _engine = WhisperEngine(MODEL_PATH, quantize=QUANTIZE)
    return _engine


def _infer(mono):
    with _engine_lock:
        return engine().transcribe([mono])[0]


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Load the model in the background so /health answers "loading" at once, then "ready".
    if os.environ.get("ASR_PRELOAD", "1") == "1":
        threading.Thread(target=engine, daemon=True).start()
    yield


app = FastAPI(title="Amharic ASR (teaching service)", version=MODEL_VERSION, lifespan=lifespan)


def _write(event: dict) -> None:
    line = json.dumps(event, ensure_ascii=False)
    with _lock:
        with _req_file.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    log.info(line)


@app.get("/health")
def health():
    return {"status": "ready" if _engine is not None else "loading", "model_version": MODEL_VERSION,
            "normalizer_version": NORMALIZER_VERSION, "quantized": QUANTIZE}


@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...), client: str = Form("unknown")):
    rid = uuid.uuid4().hex[:12]
    t0 = time.perf_counter()
    data = await audio.read()
    _counters["requests_total"] += 1
    g, mono = gate(data)
    event = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "request_id": rid, "model_version": MODEL_VERSION,
             "normalizer_version": NORMALIZER_VERSION, "client": client[:32], **g.features,
             "gate_ok": g.ok, "gate_issues": g.issues, "gate_warnings": g.warnings}
    if not g.ok and not (SKIP_GATE and mono is not None):
        _counters["rejected_total"] += 1
        event.update(status="rejected", latency_ms=round((time.perf_counter() - t0) * 1000, 1))
        _write(event)
        raise HTTPException(422, detail={"request_id": rid, "model_version": MODEL_VERSION,
                                         "issues": g.issues, "warnings": g.warnings})
    if MAX_WAITING and _state["waiting"] >= MAX_WAITING:
        # Backpressure: fail fast and honestly instead of making a clinician wait 30+ s.
        _counters["busy_total"] += 1
        event.update(status="busy", latency_ms=round((time.perf_counter() - t0) * 1000, 1))
        _write(event)
        raise HTTPException(503, detail={"request_id": rid, "model_version": MODEL_VERSION,
                                         "issues": ["server busy, retry in a few seconds"]},
                            headers={"Retry-After": "3"})
    try:
        _state["waiting"] += 1
        try:
            text = await run_in_threadpool(_infer, mono)
        finally:
            _state["waiting"] -= 1
    except Exception as e:
        _counters["errors_total"] += 1
        event.update(status="error", error=type(e).__name__)
        _write(event)
        raise HTTPException(500, detail={"request_id": rid, "model_version": MODEL_VERSION})
    latency = time.perf_counter() - t0
    _latencies.append(latency)
    empty = not text.strip()
    _counters["empty_total"] += empty
    event.update(status="ok", latency_ms=round(latency * 1000, 1), n_chars=len(text), empty=empty)
    if LOG_TEXT:
        event["text"] = text  # WRONG PATH: transcripts can contain names, diagnoses, phone numbers
    _write(event)
    return {"request_id": rid, "text": text, "model_version": MODEL_VERSION,
            "normalizer_version": NORMALIZER_VERSION, "latency_ms": round(latency * 1000, 1),
            "warnings": g.warnings}


@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    """Prometheus text format, written by hand (no extra dependency)."""
    lines = []
    for k, v in _counters.items():
        lines += [f"# TYPE asr_{k} counter", f'asr_{k}{{model_version="{MODEL_VERSION}"}} {v}']
    lat = sorted(_latencies[-1000:])
    if lat:
        for q in (0.5, 0.95):
            lines.append(f'asr_latency_seconds{{quantile="{q}",model_version="{MODEL_VERSION}"}} '
                         f"{lat[min(len(lat) - 1, int(q * len(lat)))]:.3f}")
    return "\n".join(lines) + "\n"
