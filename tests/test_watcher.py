"""The watcher's rules: persistence, one message per problem, drift type, canary rollback."""
import json

import pytest

import asr.watcher as watcher


def write_logs(tmp_path, days):
    """days: list of (snr_db, edit_rate, version) per request; written as clinic traffic."""
    (tmp_path / "v1").mkdir()
    req, fb = [], []
    for i, (snr, edit, version) in enumerate(days):
        rid = f"r{i:03d}"
        req.append({"ts": f"2026-10-29T09:{i // 60:02d}:{i % 60:02d}", "request_id": rid, "model_version": version,
                    "client": "clinic-day1", "status": "ok", "empty": False, "latency_ms": 1500, "snr_db": snr})
        fb.append({"request_id": rid, "model_version": version, "clip_id": f"c{i}", "edit_rate": edit, "day": 1})
    (tmp_path / "v1" / "requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in req), encoding="utf-8")
    (tmp_path / "feedback.jsonl").write_text("".join(json.dumps(r) + "\n" for r in fb), encoding="utf-8")


@pytest.fixture
def logs(tmp_path, monkeypatch):
    monkeypatch.setattr(watcher, "LOG_DIR", tmp_path)
    monkeypatch.setattr(watcher, "QUEUE", tmp_path / "queue.csv")
    return tmp_path


CHAMP = "tiny-600hr"
GOOD = [(42.0, 0.10, CHAMP)] * 20
PHONES = [(4.0, 0.80, CHAMP)] * 20
OUTBREAK = [(42.0, 0.40, CHAMP)] * 20


def titles(msgs):
    return [(m["severity"], m["rule"]) for m in msgs]


def test_quiet_traffic_sends_nothing(logs):
    write_logs(logs, GOOD * 2)
    assert watcher.Watcher(mode="persistent", dry=True).step() == []


def test_data_drift_fires_once_then_resolves(logs):
    write_logs(logs, GOOD + PHONES + PHONES + GOOD)
    msgs = watcher.Watcher(mode="persistent", dry=True).step()
    assert titles(msgs) == [("P2", "channel_shift"), ("RESOLVED", "channel_shift")]
    assert any("annotation queue" in a for a in msgs[0]["actions"])


def test_one_bad_window_is_not_an_alert(logs):
    write_logs(logs, GOOD + PHONES[:10] + GOOD)
    assert watcher.Watcher(mode="persistent", dry=True).step() == []


def test_more_edits_with_unchanged_audio_is_concept_drift(logs):
    write_logs(logs, GOOD + OUTBREAK)
    assert titles(watcher.Watcher(mode="persistent", dry=True).step()) == [("P2", "notes_changed")]


def test_more_edits_with_bad_audio_is_data_drift(logs):
    write_logs(logs, GOOD + PHONES)
    assert titles(watcher.Watcher(mode="persistent", dry=True).step()) == [("P2", "channel_shift")]


def test_naive_mode_floods_the_inbox(logs):
    write_logs(logs, GOOD + PHONES + PHONES)
    naive = watcher.Watcher(mode="naive", dry=True).step()
    persistent = watcher.Watcher(mode="persistent", dry=True).step()
    assert len(naive) >= 40 and len(persistent) == 1


def test_canary_worse_rolls_back(logs, monkeypatch):
    calls = []

    class Resp:
        def json(self):
            return {"canary_percent": 0.0}

    monkeypatch.setattr(watcher.httpx, "post", lambda url, **kw: calls.append((url, kw)) or Resp())
    traffic = [(42.0, 0.10, CHAMP) if i % 2 else (42.0, 0.25, "tiny-600hr-int8") for i in range(40)]
    write_logs(logs, traffic)
    msgs = watcher.Watcher(mode="persistent", dry=False).step()
    assert titles(msgs) == [("P1", "canary_worse")]
    assert calls and calls[0][0].endswith("/admin/canary") and calls[0][1]["data"] == {"percent": 0}
    assert "rolled back" in msgs[0]["actions"][0]


def test_alerts_never_contain_text(logs):
    write_logs(logs, GOOD + PHONES + PHONES)
    watcher.Watcher(mode="persistent", dry=False).step()
    sent = (logs / "alerts.jsonl").read_text(encoding="utf-8")
    assert "text" not in json.loads(sent.splitlines()[0])
