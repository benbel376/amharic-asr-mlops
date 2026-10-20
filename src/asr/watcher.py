"""Automated response to drift: detect -> notify -> act safely. People decide the rest.

    uvicorn asr.watcher:app --port 8060          # started by scripts/start_server.sh; inbox at http://127.0.0.1:8060
    python -m asr.watcher --replay               # run the rules once over logs/ and print what would be sent
    python -m asr.watcher --replay --mode naive  # the same logs, alerting on every bad request (episode 6)

Every few seconds the watcher reads the request log and the clinician feedback (no audio, no text),
cuts the clinic traffic into windows of N requests, computes the signals per window and applies the
alert rules in configs/params.yaml (`alerts:`). A rule fires only when it holds for `windows`
consecutive windows, sends one message per problem (and one when it clears), and runs the actions
listed for it. Allowed actions are safe and reversible:

    annotation_queue   high-edit clips go to data/annotation/queue.csv (clip id only)
    draft_gate3        a draft Gate 3 review record with the evidence, in logs/gate3/
    rollback_canary    set the gateway's canary share to 0 (all traffic back to the champion)

Promoting, retraining and deploying are never done here: they need a person and a Gate 2 record.
Messages go to logs/alerts.jsonl (shown in the inbox page). If TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID
are set, they are also sent to that Telegram group; messages that fail to send are kept and retried.
"""
from __future__ import annotations

import argparse
import asyncio
import html
import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import logging

import httpx
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
CFG = yaml.safe_load((ROOT / "configs" / "params.yaml").read_text(encoding="utf-8"))
W = CFG["watcher"]
RULES = CFG["alerts"]
LOG_DIR = Path(os.environ.get("ASR_LOG_DIR", str(ROOT / "logs"))).resolve()
GATEWAY = os.environ.get("GATEWAY_URL", "http://127.0.0.1:8000")
QUEUE = ROOT / "data" / "annotation" / "queue.csv"
log = logging.getLogger("asr.watcher")


# ---------------------------------------------------------------- signals

def _read(name: str) -> pd.DataFrame:
    frames = []
    for f in sorted(LOG_DIR.glob(f"**/{name}")):
        rows = [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines() if x.strip()]
        if rows:
            frames.append(pd.DataFrame(rows))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def clinic_traffic() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Clinic requests in arrival order, and the clinician feedback joined to them."""
    req = _read("requests.jsonl")
    if req.empty:
        return req, req
    if "client" in req:  # clinic traffic only: not load tests, shadow copies or debugging calls
        req = req[req.client.astype(str).str.startswith("clinic")]
    req = req.sort_values("ts", kind="stable").reset_index(drop=True)
    fb = _read("feedback.jsonl")
    return req, fb


def window_signals(w: pd.DataFrame, fb: pd.DataFrame) -> dict:
    ok = w[w.status == "ok"]
    s = {"requests": len(w),
         "rejected": round(float((w.status == "rejected").mean()), 3),
         "empty_rate": round(float(ok["empty"].mean()), 3) if len(ok) else None,
         "snr_db_median": round(float(w.snr_db.median()), 1) if "snr_db" in w and w.snr_db.notna().any() else None,
         "p95_ms": round(float(ok.latency_ms.quantile(0.95))) if len(ok) else None,
         "edit_rate": None, "edit_rate_gap": None, "versions": {}}
    if not fb.empty:
        e = fb[fb.request_id.isin(w.request_id)]
        if len(e):
            s["edit_rate"] = round(float(e.edit_rate.mean()), 3)
            per = e.groupby("model_version").edit_rate.agg(["mean", "size"])
            s["versions"] = {v: {"edit_rate": round(float(r["mean"]), 3), "n": int(r["size"])} for v, r in per.iterrows()}
    champ = CFG["models"]["champion"]["id"]
    cands = [v for v in s["versions"] if v != champ]
    if champ in s["versions"] and cands:
        c = s["versions"][cands[0]]
        if c["n"] >= W["min_per_version"] and s["versions"][champ]["n"] >= W["min_per_version"]:
            s["edit_rate_gap"] = round(c["edit_rate"] - s["versions"][champ]["edit_rate"], 3)
            s["candidate"] = cands[0]
    return s


def windows(req: pd.DataFrame, fb: pd.DataFrame, size: int, grace_s: float = 20) -> list[dict]:
    """Complete windows only: N requests, and every answered request already has its correction
    (or the window is older than grace_s, for requests nobody corrected)."""
    out = []
    have = set(fb.request_id) if not fb.empty else set()
    for start in range(0, len(req) - size + 1, size):
        w = req.iloc[start:start + size]
        waiting = [r for r in w[w.status == "ok"].request_id if r not in have]
        age = time.time() - time.mktime(time.strptime(str(w.ts.iloc[-1]), "%Y-%m-%dT%H:%M:%S"))
        if waiting and age < grace_s:
            break
        out.append({"index": start // size + 1, "span": f"{start + 1}-{start + size}",
                    "request_ids": list(w.request_id), "rows": w, **window_signals(w, fb)})
    return out


def pool(group: list[dict], fb: pd.DataFrame) -> dict:
    rows = pd.concat([w["rows"] for w in group])
    return {"index": group[-1]["index"], "span": f"{group[0]['span'].split('-')[0]}-{group[-1]['span'].split('-')[1]}",
            "request_ids": list(rows.request_id), "rows": rows, **window_signals(rows, fb)}


def holds(rule: dict, sig: dict) -> bool:
    for cond in [rule] + ([rule["and"]] if "and" in rule else []):
        v = sig.get(cond["signal"])
        if v is None:
            return False
        if "above" in cond and not v > cond["above"]:
            return False
        if "below" in cond and not v < cond["below"]:
            return False
    return True


# ---------------------------------------------------------------- actions

def _show(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)



def act(action: str, rule: dict, evidence: list[dict], fb: pd.DataFrame, dry: bool) -> str:
    if action == "annotation_queue":
        ids = {r for w in evidence for r in w["request_ids"]}
        e = fb[fb.request_id.isin(ids) & (fb.edit_rate > CFG["monitoring"]["annotation_queue_edit_rate"])] if not fb.empty else fb
        if dry or e.empty:
            return f"{len(e)} clips to the annotation queue"
        cols = [c for c in ("clip_id", "request_id", "model_version", "edit_rate", "day") if c in e]
        rows = e[cols].assign(reason=rule["name"])
        old = pd.read_csv(QUEUE, dtype=str) if QUEUE.exists() and QUEUE.stat().st_size else pd.DataFrame()
        QUEUE.parent.mkdir(parents=True, exist_ok=True)
        merged = pd.concat([old, rows.astype(str)], ignore_index=True).drop_duplicates("request_id")
        merged.to_csv(QUEUE, index=False)
        return f"{len(rows)} clips to the annotation queue ({_show(QUEUE)})"
    if action == "draft_gate3":
        path = LOG_DIR / "gate3" / f"gate3_DRAFT_{time.strftime('%Y-%m-%d')}_{rule['name']}.md"
        if not dry:
            path.parent.mkdir(parents=True, exist_ok=True)
            lines = [f"# Gate 3 review (DRAFT, opened automatically by the watcher)", "",
                     f"- Trigger: alert `{rule['name']}` ({rule['kind']}), severity {rule['severity']}",
                     f"- Opened: {time.strftime('%Y-%m-%d %H:%M')} EAT", f"- Owner: {rule['owner']}",
                     "- Decision (keep / retrain / restrict / retire): **pending, a person decides**", "",
                     "| window | requests | SNR median dB | edit rate | empty | rejected |", "|---|---|---|---|---|---|"]
            lines += [f"| {w['span']} | {w['requests']} | {w['snr_db_median']} | {w['edit_rate']} | "
                      f"{w['empty_rate']} | {w['rejected']} |" for w in evidence]
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return f"draft Gate 3 record {_show(path)}"
    if action == "rollback_canary":
        if dry:
            return "canary rolled back to 0% (dry run)"
        try:
            r = httpx.post(f"{GATEWAY}/admin/canary", data={"percent": 0}, timeout=5)
            return f"canary rolled back: gateway now sends {r.json()['canary_percent']:.0f}% to the candidate"
        except Exception as e:  # the alert still goes out; a person rolls back by hand
            return f"ROLLBACK FAILED ({type(e).__name__}): roll back by hand"
    return f"unknown action {action} (ignored)"


# ---------------------------------------------------------------- the watcher

class Watcher:
    def __init__(self, mode: str | None = None, dry: bool = False):
        self.mode = mode or W["mode"]
        self.dry = dry
        self.seen_windows = 0
        self.seen_requests = 0
        self.active: dict[str, int] = {}  # rule name -> window index when it fired
        self.messages: list[dict] = []

    def _send(self, msg: dict) -> None:
        msg["ts"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self.messages.append(msg)
        if self.dry:
            return
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        with (LOG_DIR / "alerts.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(msg, ensure_ascii=False) + "\n")
        telegram(msg)

    def step(self) -> list[dict]:
        """Look at new traffic once; return the messages sent."""
        before = len(self.messages)
        req, fb = clinic_traffic()
        if req.empty:
            return []
        if self.mode == "naive":
            self._naive(req, fb)
        else:
            wins = windows(req, fb, W["window"])
            for i in range(self.seen_windows, len(wins)):
                self._evaluate(wins[: i + 1], fb)
            self.seen_windows = len(wins)
        return self.messages[before:]

    def _naive(self, req: pd.DataFrame, fb: pd.DataFrame) -> None:
        """WRONG PATH: one message for every request that crosses any threshold."""
        new = req.iloc[self.seen_requests:]
        fbi = fb.set_index("request_id") if not fb.empty else None
        for _, r in new.iterrows():
            e = fbi.edit_rate.get(r.request_id) if fbi is not None else None
            if r.get("snr_db") is not None and r.snr_db < CFG["monitoring"]["min_snr_db_median"]:
                self._send({"severity": "ALERT", "rule": "snr", "title": f"Low SNR {r.snr_db:.1f} dB", "body": [], "actions": []})
            if e is not None and e > CFG["monitoring"]["max_edit_rate"]:
                self._send({"severity": "ALERT", "rule": "edits", "title": f"Edit rate {e:.2f}", "body": [], "actions": []})
            if r.status == "rejected":
                self._send({"severity": "ALERT", "rule": "rejected", "title": "Request rejected", "body": [], "actions": []})
        self.seen_requests = len(req)

    def _evaluate(self, wins: list[dict], fb: pd.DataFrame) -> None:
        last = wins[-1]
        if last["requests"] - int(last["rejected"] * last["requests"]) < W["min_requests"]:
            return  # too few usable requests to judge
        for rule in RULES:
            n, span = rule.get("windows", 1), rule.get("span", 1)
            # span > 1: pool the last `span` windows before computing the signal (small samples, e.g. a canary)
            pooled = [w if span == 1 else pool(wins[max(0, i - span + 1): i + 1], fb)
                      for i, w in enumerate(wins)]
            recent = pooled[-n:]
            firing = len(recent) == n and all(holds(rule, w) for w in recent)
            name = rule["name"]
            if firing and name not in self.active:
                self.active[name] = last["index"]
                done = []
                for a in rule.get("actions", []):
                    try:
                        done.append(act(a, rule, recent, fb, self.dry))
                    except Exception as e:  # a failed action must not swallow the alert
                        done.append(f"{a} FAILED ({type(e).__name__}: {e})")
                v = recent[-1]
                body = [f"{rule['signal']} = {v.get(rule['signal'])} for {n} windows in a row "
                        f"(threshold {'>' if 'above' in rule else '<'} {rule.get('above', rule.get('below'))})",
                        f"requests {recent[0]['span'].split('-')[0]}-{v['span'].split('-')[1]}: SNR median "
                        f"{', '.join(str(w['snr_db_median']) for w in recent)} dB; edit rate "
                        f"{', '.join(str(w['edit_rate']) for w in recent)}"]
                if v.get("candidate"):
                    body.append(f"{v['candidate']} vs champion edit rate gap: "
                                f"{', '.join(str(w['edit_rate_gap']) for w in recent)}")
                body.append(f"Owner: {rule['owner']}. Next step: {rule['next']}")
                self._send({"severity": rule["severity"], "rule": name, "title": f"{rule['kind']}: {rule['title']}",
                            "body": body, "actions": done})
            elif not firing and name in self.active and not holds(rule, pooled[-1]) and rule.get("notify_resolved", True):
                del self.active[name]
                self._send({"severity": "RESOLVED", "rule": name, "title": f"{rule['title']}: back to normal",
                            "body": [f"window {last['span']}: {rule['signal']} = {last.get(rule['signal'])}"],
                            "actions": []})


def telegram(msg: dict) -> None:
    """Optional: forward to a Telegram group. The token is a secret (.env, never in git)."""
    token, chat = os.environ.get("TELEGRAM_BOT_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    if not (token and chat):
        return
    outbox = LOG_DIR / "telegram_outbox.jsonl"
    pending = [json.loads(x) for x in outbox.read_text(encoding="utf-8").splitlines()] if outbox.exists() else []
    pending.append(msg)
    left = []
    for m in pending:
        text = f"[{m['severity']}] {m['title']}\n" + "\n".join(m["body"] + [f"Done: {a}" for a in m["actions"]])
        try:
            httpx.post(f"https://api.telegram.org/bot{token}/sendMessage", data={"chat_id": chat, "text": text},
                       timeout=10).raise_for_status()
        except Exception:
            left.append(m)  # no internet: keep it and try again next time
    outbox.write_text("".join(json.dumps(m, ensure_ascii=False) + "\n" for m in left), encoding="utf-8")


# ---------------------------------------------------------------- inbox page

COLOURS = {"P1": "#EA7201", "P2": "#02C2DE", "P3": "#979797", "RESOLVED": "#2E9E5B", "ALERT": "#EA7201"}


def inbox_html() -> str:
    path = LOG_DIR / "alerts.jsonl"
    msgs = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()] if path.exists() else []
    items = []
    for m in reversed(msgs):
        c = COLOURS.get(m["severity"], "#979797")
        body = "".join(f"<div>{html.escape(b)}</div>" for b in m["body"])
        acts = "".join(f"<div class=a>&#10003; {html.escape(a)}</div>" for a in m["actions"])
        items.append(f"<div class=m style='border-left-color:{c}'><div class=h><b style='color:{c}'>{m['severity']}</b> "
                     f"{html.escape(m['title'])}<span>{m['ts'][11:16]}</span></div>{body}{acts}</div>")
    return ("<!doctype html><meta charset=utf-8><meta http-equiv=refresh content=5><title>ASR alerts</title>"
            "<style>body{margin:0;font-family:Segoe UI,Roboto,sans-serif;background:#E7EBF0;color:#1C1C1C}"
            "header{background:#1C1C1C;color:#fff;padding:14px 20px;font-size:17px}header small{color:#979797;margin-left:8px}"
            ".m{background:#fff;margin:10px 16px;padding:10px 14px;border-radius:10px;border-left:5px solid;max-width:760px;"
            "box-shadow:0 1px 2px #0002;font-size:14px;line-height:1.45}.h{margin-bottom:4px}.h span{float:right;color:#979797}"
            ".a{color:#0189A0}p{margin:16px}</style>"
            f"<header>EAII ASR alerts <small>{len(msgs)} messages &middot; mock Telegram group &middot; refreshes every 5 s</small></header>"
            + ("".join(items) or "<p>No alerts yet.</p>"))


_watcher = Watcher()


async def _loop():
    while True:
        try:
            await asyncio.to_thread(_watcher.step)
        except Exception as e:  # never let one bad log line stop the watcher
            log.error("watcher error: %s: %s", type(e).__name__, e)
        await asyncio.sleep(W["interval_s"])


try:
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse

    @asynccontextmanager
    async def lifespan(_app):
        task = asyncio.create_task(_loop())
        yield
        task.cancel()

    app = FastAPI(title="ASR watcher", lifespan=lifespan)

    @app.get("/", response_class=HTMLResponse)
    def index():
        return inbox_html()

    @app.get("/health")
    def health():
        return {"status": "ready", "mode": _watcher.mode, "active_alerts": sorted(_watcher.active)}
except ImportError:  # pragma: no cover
    app = None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", action="store_true", help="run the rules once over the logs; send nothing")
    ap.add_argument("--mode", choices=["naive", "persistent"])
    a = ap.parse_args()
    w = Watcher(mode=a.mode, dry=True)
    msgs = w.step()
    if w.mode == "naive":
        by = pd.Series([m["rule"] for m in msgs]).value_counts().to_dict() if msgs else {}
        print(f"naive mode: {len(msgs)} messages would be sent {by}")
        return
    for m in msgs:
        print(f"[{m['severity']}] {m['title']}")
        for b in m["body"]:
            print(f"    {b}")
        for x in m["actions"]:
            print(f"    done: {x}")
    print(f"{len(msgs)} messages")


if __name__ == "__main__":
    main()
