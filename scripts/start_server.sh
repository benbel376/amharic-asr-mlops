#!/usr/bin/env bash
# Start the simulated production server WITHOUT Docker (five uvicorn processes).
#   bash scripts/start_server.sh            # champion :8001, candidate :8002, gateway :8000, monitor :8050, watcher :8060
#   bash scripts/start_server.sh stop       # also stops the episode-7 debug service on :8003
# Same services and ports as docker-compose.yml (except champion/candidate are internal there).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe; [ -x "$PY" ] || PY=.venv/bin/python
export PYTHONPATH=src PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1
if [ "${1:-}" = "stop" ]; then
  for f in logs/*.pid; do [ -f "$f" ] && kill "$(cat "$f")" 2>/dev/null || true; done
  # Windows fallback: kill anything still listening on our ports
  command -v powershell >/dev/null && powershell -NoProfile -Command \
    "foreach (\$p in 8000,8001,8002,8003,8050,8060) { Get-NetTCPConnection -LocalPort \$p -State Listen -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id \$_.OwningProcess -Force } }" || true
  echo "stopped"; exit 0
fi
mkdir -p logs/v1 logs/v2
# Each start is a new "deployment": keep the old logs (renamed, so the monitor no longer reads them).
ts=$(date +%Y%m%d-%H%M%S)
for f in logs/v1/requests.jsonl logs/v2/requests.jsonl logs/feedback.jsonl logs/alerts.jsonl; do
  if [ -s "$f" ]; then mv "$f" "${f%.jsonl}-$ts.jsonl"; fi
done
if [ -s data/annotation/queue.csv ]; then mv data/annotation/queue.csv "data/annotation/queue-$ts.csv"; fi
if [ -d logs/gate3 ]; then mv logs/gate3 "logs/gate3-$ts"; fi
rm -rf logs/naive   # episode 7 debug log holds transcripts: never keep it
CANDIDATE_PATH=${CANDIDATE_PATH:-models/tiny-stage2}
CANDIDATE_VERSION=${CANDIDATE_VERSION:-tiny-stage2}
ASR_MAX_WAITING=${ASR_MAX_WAITING:-0} ASR_MODEL_PATH=models/tiny-600hr ASR_MODEL_VERSION=tiny-600hr ASR_LOG_DIR=logs/v1 \
  "$PY" -m uvicorn asr.service:app --port 8001 > logs/champion.out 2>&1 & echo $! > logs/champion.pid
ASR_MAX_WAITING=${ASR_MAX_WAITING:-0} ASR_QUANTIZE=${CANDIDATE_QUANTIZE:-0} ASR_MODEL_PATH=$CANDIDATE_PATH \
  ASR_MODEL_VERSION=$CANDIDATE_VERSION ASR_LOG_DIR=logs/v2 \
  "$PY" -m uvicorn asr.service:app --port 8002 > logs/candidate.out 2>&1 & echo $! > logs/candidate.pid
CANARY_PERCENT=${CANARY_PERCENT:-0} "$PY" -m uvicorn asr.gateway:app --port 8000 > logs/gateway.out 2>&1 & echo $! > logs/gateway.pid
ASR_LOG_DIR=logs "$PY" -m uvicorn asr.monitor:app --port 8050 > logs/monitor.out 2>&1 & echo $! > logs/monitor.pid
ASR_LOG_DIR=logs GATEWAY_URL=http://127.0.0.1:8000 "$PY" -m uvicorn asr.watcher:app --port 8060 > logs/watcher.out 2>&1 & echo $! > logs/watcher.pid
for i in $(seq 1 30); do
  if curl -s localhost:8000/health 2>/dev/null | grep -q '"status":"ready".*"status":"ready"'; then
    echo "ready: gateway http://127.0.0.1:8000  monitor http://127.0.0.1:8050  alerts http://127.0.0.1:8060"; exit 0; fi
  sleep 2
done
echo "not ready after 60 s: see logs/*.out"; exit 1
