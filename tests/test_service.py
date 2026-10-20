"""Service contract and privacy tests. A fake engine keeps them fast (no model download)."""
import io
import json

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

SECRET = "አበበ በቀለ 0911223344"  # synthetic: a made-up name and phone number the model "heard"


class FakeEngine:
    def transcribe(self, batch):
        return [SECRET for _ in batch]


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("ASR_LOG_DIR", str(tmp_path))
    monkeypatch.setenv("ASR_PRELOAD", "0")
    import importlib

    import asr.service as svc
    importlib.reload(svc)
    svc._engine = FakeEngine()
    return TestClient(svc.app), tmp_path


def wav(x, sr=16000):
    buf = io.BytesIO()
    sf.write(buf, x, sr, format="WAV")
    return buf.getvalue()


def speech():
    return (0.1 * np.random.default_rng(0).standard_normal(32000)).astype("float32")


def test_every_response_has_model_version(client):
    c, _ = client
    r = c.post("/transcribe", files={"audio": ("a.wav", wav(speech()), "audio/wav")})
    assert r.status_code == 200
    body = r.json()
    assert body["model_version"] and body["normalizer_version"] == "am-norm-v1" and body["request_id"]


def test_logs_never_contain_transcript(client):
    c, logdir = client
    c.post("/transcribe", files={"audio": ("a.wav", wav(speech()), "audio/wav")})
    log = (logdir / "requests.jsonl").read_text(encoding="utf-8")
    assert SECRET.split()[-1] not in log and SECRET.split()[0] not in log
    ev = json.loads(log.splitlines()[-1])
    assert "text" not in ev and ev["n_chars"] == len(SECRET) and ev["model_version"]


def test_silence_rejected_with_reason(client):
    c, _ = client
    r = c.post("/transcribe", files={"audio": ("s.wav", wav(np.zeros(32000, "float32")), "audio/wav")})
    assert r.status_code == 422 and "silent" in json.dumps(r.json(), ensure_ascii=False)


def test_health_and_metrics(client):
    c, _ = client
    assert c.get("/health").json()["model_version"]
    c.post("/transcribe", files={"audio": ("a.wav", wav(speech()), "audio/wav")})
    m = c.get("/metrics").text
    assert "asr_requests_total" in m and "asr_latency_seconds" in m
