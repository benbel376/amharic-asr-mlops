# Facilitator notes: Amharic ASR session (Thu 29 Oct, 09:30-12:30)

Facilitators: Abenezer and Biniyam. Participants: EAII NLP team (they know ASR better than most audiences; do not teach CTC or Whisper; ask them to explain when it helps the room).

## Timeline

| Clock | Block | Who leads | Notes |
|---|---|---|---|
| 09:30-09:45 | Intro (setting, six beats, today's scenario) | Biniyam | Show the OpenMRS Speak button (screen recording from our EMR assistant prototype) as "where this ends up". 3 slides max |
| 09:45-09:55 | Ep 0 setup | both | Most installs done on Tuesday? If not: USB kit (`.venv` cannot be copied; copy uv cache + `data/raw` + `models`) |
| 09:55-10:10 | Ep 1 normalisation | Abenezer | |
| 10:10-10:20 | Ep 2 leakage | Abenezer | |
| 10:20-10:35 | Ep 3 audio gate | Biniyam | |
| 10:35-10:55 | Ep 4 champion vs candidate | Biniyam | MLflow UI here |
| 10:55-11:05 | break | | Start `bash scripts/start_server.sh` before the break |
| 11:05-11:20 | Ep 5 load | Abenezer | |
| 11:20-11:40 | Ep 6 monitoring + feedback | Biniyam | |
| 11:40-11:55 | Ep 7 canary, rollback, privacy | Abenezer | |
| 11:55-12:25 | Discussion | both | Questions below |
| 12:25-12:30 | Wrap-up, take-home (ep 8, 9) | | |

Hands-on total: about 2 h 5 min including the PAUSEs. If behind by 10 min at the break: skip Ep 2 commands (show the two numbers) and run Ep 7's canary from the pre-computed table. Episode 8 becomes take-home.

## Per episode: what to say, expected output, fallback

**Ep 0.** Expected `31 passed`. Measured here: install 3 min from a fast mirror (1.1 GB, 113 packages), download 420 MB. If PyPI is slow, use the mirror or a local uv cache from USB. If `python` prints `?????` for Amharic: they forgot `source scripts/env.sh` (PYTHONIOENCODING).

**Ep 1.** Say: "Your paper numbers and your production numbers must be scored the same way. The version string is the contract." Ask: "Which of these differences would a clinician care about?" Point out the ሦ/ሶ, ዓ/አ variants come from FLEURS itself. Discussion hook: should homophone folding be in the normaliser or in the transcription guideline? (Both: guideline lets annotators ignore it, scorer folds it.) The aside (2 min) re-runs the champion with `--max-new-tokens 225`, the lab's own earlier setting: 11 of 40 outputs are cut and CER goes from 12.4% to 18.7%. Every number this lab published before 20 Oct 2026 (WER 40.5 / CER 18.7) had that bug. Both runs sit side by side in MLflow (`tiny-600hr` and `tiny-600hr-t225`, with `max_new_tokens` as a parameter): a good "why tracking matters" moment. Ask: what else in your decoding config could silently cap quality? (beam size, length penalty, chunking of long audio.)
Fallback: `results/precomputed/tiny-600hr.json` and `_hyps.csv`.

**Ep 2.** Say: "FLEURS is well-built and still leaks if you split it naively." Ask about their own corpora: speaker ids, same prompt read by many people, same patient on two days. The constant gender column is real; ask what they would put in the dataset card.

**Ep 3.** Say: "The model did not fail. It did what it was trained to do: produce Amharic text." Ask: reject or warn? Who sees the reason? Note honestly that our first gate threshold rejected real quiet speakers (18 of 60 requests): gates need calibration and a monitored rejection rate. Room noise and hum still pass: a VAD or a "no speech" probability is the next step (Whisper's `no_speech_prob`, Silero VAD as in our EMR assistant prototype).

**Ep 4.** Say: "Write the rule before you look." Ask: is 0.2 WER points on 40 clips a real difference? (No: 7 min of audio.) Phone-channel CER is above 80% and int8 passes 100%: on noisy audio the models repeat themselves until the 440-token limit (the `cut` column), and insertions count as errors. Who signs Gate 2? Discuss the int8 result: dynamic quantisation hurt this tiny model badly and did not speed it up on this CPU; on larger models and ONNX/CTranslate2 int8 it often works. Measure, do not assume.
Fallback: `results/precomputed/*.json`, then `python scripts/promote.py` works from those.

**Ep 5.** Expected (24-thread laptop): 1 user p50 2.1-2.7 s; 30 users naive 17 ok / 43 timeouts, p50 14.7-16.3 s; with `ASR_MAX_WAITING=4` 8 ok / 52 fast 503, p95 7.4-7.5 s, 14-15 s wall. The ok/error counts repeated exactly in three runs; latencies move by up to half a second with whatever else the laptop is doing. Slower laptops: more timeouts, same story. Say: "Latency under load is a property of the deployment, not the model." Ask: what does the app do with a 503? (Show "busy, try again", keep the typed note option.) Capacity: one replica about 0.4-0.5 requests/s; 30 clinicians who each dictate once a minute is 0.5 requests/s at peak.
Fallback: `results/precomputed/load_30users_naive.json`, `load_30users_backpressure.json`.

**Ep 6.** Takes about 2 min to run. Edit rate 0.115 on day 1, 0.792 on the new phones; 40 clips queued. The p95 latency also rises (about 3.3-4.0 s to 4.3-4.5 s): the model loops on noise until the token limit, so it is slower exactly when it is wrong. Say: "In production you rarely have labels. You do have the microphone, and the clinician's corrections." Ask: what else is label-free? (empty rate, output length per second of audio, language-id, confidence.) Why store only the edit rate and not the corrected text in the log? Where does the corrected text go instead? (Annotation store with consent and access control, PDPP 1321/2024.)
Fallback: open `results/precomputed/monitor_after_clinic.html`; logs in `results/precomputed/clinic_*.jsonl`.

**Ep 7.** Canary numbers (40 requests, 50/50, 20 each): champion edit rate 0.111, int8 0.228; latency about equal (p50 2.0-2.4 s each). The gateway hashes the audio, so the split and the numbers repeat exactly. Neither row crosses the 0.30 edit-rate alert: the decision needs a person comparing two numbers. Say: "Rollback is a number, a named person and one command. Practise it before you need it." For privacy: show the naive log line with text, then the test that fails the build. The canary admin endpoint has no authentication in the lab: say so.
Fallback: `results/precomputed/canary_*.jsonl`.

## Discussion questions (30 min)

1. Your production model will be Ethio-ASR (2.4 GB). Which of today's checks change, and which stay exactly the same?
2. Where should the audio gate run: in the browser, at the gateway, in the model service? (Hint from Yoni's lesson: per-record rules can run anywhere; keep one rules file.)
3. Clinician edits are biased labels: tired clinicians accept errors. How would you audit the edit-rate signal? (F07, F10)
4. Code-switching: medical terms in English inside Amharic. None of the off-the-shelf models writes them in Latin script. What evaluation set would you build first, and with whose consent?
5. What would make you roll back without waiting for a Gate 3 review?

## Prepare before the day

- [ ] Run Ep 0-7 once on a participant-like laptop (8 GB RAM, 4 cores) and note timings.
- [ ] USB: `data/raw/`, `models/`, a uv cache (`uv cache dir`) with the 113 wheels, this folder.
- [ ] Decide: hand participants this folder as a git repo (GitHub) or a zip? (CI only runs with a GitHub repo.)
- [ ] Optional: Docker Desktop on one facilitator laptop to show `docker compose -p eaii-asr up -d --build` (tested 20 Oct: 3 min build, 2.0 GB image; stop `start_server.sh` first, same ports). Participants never need Docker.
- [ ] Optional: a real server URL for the final demo: `python scripts/load_test.py --url http://<server>:8000`.

## Integrating into the workshop repo

GitHub runs workflows only from the repository's top-level `.github/workflows/`. If this folder becomes, say, `09-asr/` in the workshop repo, `3-asr-project/.github/` is ignored there. Put this file at `.github/workflows/asr-lab.yml` in the repo root (change `09-asr` to the real folder name); it is the lab's `ci.yml` with `working-directory` and path filters added:

```yaml
name: asr-lab
on:
  push: {paths: ["09-asr/**", ".github/workflows/asr-lab.yml"]}
  pull_request: {paths: ["09-asr/**", ".github/workflows/asr-lab.yml"]}
  workflow_dispatch:
defaults:
  run: {working-directory: 09-asr}
jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
      - name: Install (lock file; pywin32 is Windows-only, torch from the CPU index)
        run: |
          grep -v '^pywin32==' requirements.lock > /tmp/requirements-linux.lock
          uv venv --python 3.12
          uv pip install -r /tmp/requirements-linux.lock --torch-backend cpu
      - name: Unit tests
        run: .venv/bin/python -m pytest -q
  gate2-evidence:            # manual run, or a PR labelled "evaluate". RED with the shipped candidate (episode 4).
    needs: unit-tests
    if: >-
      github.event_name == 'workflow_dispatch' ||
      (github.event_name == 'pull_request' && contains(github.event.pull_request.labels.*.name, 'evaluate'))
    runs-on: ubuntu-latest
    env: {PYTHONPATH: src, HF_HUB_DISABLE_TELEMETRY: "1", MLFLOW_DISABLE_TELEMETRY: "true"}
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
      - name: Install
        run: |
          grep -v '^pywin32==' requirements.lock > /tmp/requirements-linux.lock
          uv venv --python 3.12
          uv pip install -r /tmp/requirements-linux.lock --torch-backend cpu
      - uses: actions/cache@v4      # cache paths are relative to the repo root, not working-directory
        with: {path: "09-asr/data/raw\n09-asr/models", key: "asr-${{ hashFiles('09-asr/scripts/downloads.json') }}"}
      - run: .venv/bin/python scripts/download.py
      - run: .venv/bin/python scripts/prepare_data.py && .venv/bin/python scripts/check_data.py
      - run: |
          .venv/bin/python -m asr.evaluate --model champion
          .venv/bin/python -m asr.evaluate --model candidate
      - name: Promotion rule (fails the job on REJECT)
        run: .venv/bin/python scripts/promote.py
      - uses: actions/upload-artifact@v4
        if: always()
        with: {name: asr-gate2-evidence, path: "09-asr/results/*.json\n09-asr/results/*_hyps.csv"}
```

Then delete `09-asr/.github/` so nobody edits the copy that never runs. Each step was run by hand in a `python:3.12-slim` container; the workflow itself has not run on GitHub yet.
