# Monitoring plan (MON-01)

Labels (correct transcripts) arrive late or never, so production monitoring uses label-free signals first, then clinician edits. The watcher (`src/asr/watcher.py`, port 8060) checks these every few seconds and acts on its own within the limits below. The rules live in `configs/params.yaml` (`alerts:`); they are part of the Gate 2 record and change only through review.

## Signals and alert rules

| Signal | Source | Threshold | Persistence | Severity | Owner | Automatic action | Next step for a person |
|---|---|---|---|---|---|---|---|
| Median SNR (microphone fingerprint) | request log `snr_db` | < 15 dB (eval set 42.7 dB) | 2 windows of 10 requests | P2 | ASR lead | Notify; high-edit clips to the annotation queue; draft Gate 3 record | Find the device batch; evaluate the champion on its audio (data drift, F08) |
| Clinician edit rate with normal SNR | `feedback.jsonl` (edit rate and clip id, no text) | > 0.30 and SNR > 15 dB | 2 windows | P2 | ASR lead | Notify; annotation queue; draft Gate 3 record | Read corrected notes for new terms; plan retraining (concept drift) |
| Empty-output rate | request log `empty` | > 5% | 2 windows | P1 | Ops owner | Notify | Check the last release; roll back |
| Candidate vs champion edit rate during a canary | `feedback.jsonl` by `model_version` | gap > 0.05 | 2 checks, each pooling 2 windows | P1 | Ops owner | Notify; **roll the canary back to 0%** | Candidate returns to validation; incident record |
| Gate rejections | request log `status=rejected` | > 10% per facility per day | daily | P3 | ASR lead + facility IT | Weekly report | Microphone, browser recording settings |
| p95 latency, 503 "busy" share | request log, `/metrics` | p95 > 3 s or busy > 2% | 15 min | P2 | Ops owner | Weekly report (alert once a real server exists) | Add a replica; check CPU limits |

## What the watcher may never do

Promote a model, retrain and deploy, change thresholds, or delete data. Those need a person (Gate 2 or Gate 3).

## Channels

The lab writes alerts to `logs/alerts.jsonl` and shows them at http://127.0.0.1:8060 (a stand-in for a Telegram group). In production: a Telegram group per system, SMS for P1, email for the record. Set `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` in `.env` (never in git). Messages that cannot be sent (no internet) are kept and retried. Alerts contain counts, rates and ids only.

Never logged: audio, transcripts, names, phone numbers, MRNs (SEC-02, PDPP 1321/2024). Logs kept 90 days.
