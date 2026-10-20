# Amharic speech recognition: from notebook to production (lab guide)

Thursday 29 October 2026, 09:30-12:30. You already know ASR. Today is about everything around the model: how you score it, split data, guard the input, choose between versions, serve it, watch it and learn from clinicians' corrections.

**Scenario.** EAII and the Ministry pilot a "Speak" button in the EMR (OpenMRS): a clinician dictates a short note in Amharic, the server returns text, the clinician corrects it and saves. You own the speech service.

Every episode has the same shape:
**Setup -> Wrong path (do the obvious thing) -> The issue appears -> PAUSE: what would you do? -> Our fix -> Check -> Failure ID and rule.**
At each **PAUSE**, stop. Write your proposal on a sticky note or in the chat (2-5 minutes) before reading on.

All numbers below were verified on Windows 11 (24 logical CPUs, no GPU) on 20 October 2026. Your timings will differ; error rates should match to the third decimal.

## The framework in action

This lab walks one system through the EAII lifecycle (Practice Guide, section 1.1). Each episode shows one stage, the failure it prevents (Common Issues Catalogue) and the rule that `eaii-mlops check` verifies.

| # | Episode | Time | Lifecycle stage | Gate | Failure | Rule |
|---|---|---|---|---|---|---|
| 0 | Set up and check | 10 min | Development | | F13 | ENV-01 |
| 1 | The WER that lies (normalisation, WER vs CER) | 15 min | Validation | | F12 | REP-02 |
| 2 | The split that leaks | 5 min | Data | | F11 | DATA-02 |
| 3 | Silence that speaks (audio gate) | 15 min | Data · Deployment | | F17, F01 | DATA-03 |
| 4 | Better on average, worse for some (champion vs candidate, int8) | 20 min | Validation | Gate 2 | F14 | GOV-01 |
| 5 | Thirty clinicians press Speak (deploy, load) | 15 min | Deployment | | F16 | TST-02 |
| 6 | New phones, then an outbreak (drift and the automated response) | 25 min | Monitoring | Gate 3 trigger | F08, F05, F20 | MON-01 |
| 7 | Canary, automatic rollback, and what the logs know | 15 min | Deployment · Monitoring | | F19, F18 | GOV-01, SEC-02, LOG-01 |
| ✓ | Does this project meet the framework? | 5 min | all | Gate 2 record | | all 18 |
| 8 | Do our annotators agree? | 15 min | optional | | F10 | DATA-02 |
| 9 | Fine-tune in Colab (groups) | 30 min | take-home · Re-evaluation | | F13 | REP-01 |

Not shown hands-on: Inception and Gate 1 (the scenario above is the one-line version of a PRD) and Retirement. See the Practice Guide.

---|---|---|---|---|---|
| 0 | Set up and check | 10 min | core | F13 | ENV-01 |
| 1 | The WER that lies (normalisation, WER vs CER) | 15 min | core | F12 | REP-02 |
| 2 | The split that leaks | 10 min | core | F11 | DATA-02 |
| 3 | Silence that speaks (audio gate) | 15 min | core | F17, F01 | DATA-03 |
| 4 | Better on average, worse for some (champion vs candidate, int8) | 20 min | core | F14 | GOV-01 |
| 5 | Thirty clinicians press Speak (deploy, load) | 15 min | core | F16 | TST-02 |
| 6 | New phones arrive (label-free monitoring, clinician edits) | 20 min | core | F08, F20 | MON-01 |
| 7 | Canary, rollback, and what the logs know | 15 min | core | F19, F18 | GOV-01, SEC-02, LOG-01 |
| 8 | Do our annotators agree? | 15 min | optional | F10 | DATA-02 |
| 9 | Fine-tune in Colab (groups) | 30 min | take-home | F13 | REP-01 |

---

## Episode 0. Set up and check (10 min)

Open **Git Bash**. Use a short folder path (Windows has a 260-character path limit).

```bash
cd /c/mlops                                  # or wherever the workshop repos live
cd 3-asr-project
uv venv --python 3.12 .venv
uv pip install -r requirements.lock --python .venv/Scripts/python.exe     # 113 packages, 1.1 GB
source scripts/env.sh                        # activates .venv, sets PYTHONPATH and UTF-8 output
python scripts/download.py                   # 112 MB data + 2 x 151 MB models, checksums verified
python scripts/prepare_data.py
pytest -q
```

Expected:
```
OK  data/raw/dev.tsv  sha256 6e37bdc09086
OK  data/raw/dev.tar.gz  sha256 7d91176bd88f
OK  models/tiny-600hr  (b1n1yam/shook-tiny-amharic-600hr@7730909, Apache-2.0)
OK  models/tiny-stage2  (b1n1yam/shook-tiny-amharic-stage2-polish@4dc4d84, Apache-2.0)
ALL OK
clips: 223  sentences: 124  audio: 39.2 min  gender values: {'FEMALE': 223}
eval subset: 40 clips, 7.1 min  ->  data/eval.csv
shifted copies: 40 -> data/shift/   gate samples -> data/gate_samples/
39 passed
```
No Wi-Fi? Copy `data/raw/` and `models/` from the USB stick, then `python scripts/download.py --verify`.

Note the `gender` line. We come back to it in episode 2.

---

## Episode 1. The WER that lies (15 min)

**Setup.** Score the current model (the "champion", a Whisper-tiny Amharic fine-tune) on 40 FLEURS clips. About 1 minute.

```bash
python -m asr.evaluate --model champion
```

**Wrong path.** Report the first number you can compute: plain WER on the raw text.

**The issue appears.**
```
tiny-600hr: WER 0.356 (raw 0.493)  CER 0.124 (raw 0.156)  size 151.1 MB  RTF 0.0656
  clean          WER 0.356  CER 0.124  empty 0.00  cut  0/40
  ...
```
Raw WER is **49.3%**. Look at one clip (`results/tiny-600hr_hyps.csv`):
```
reference:  ስፔናውያን ለሦስት መቶ ዓመታት የቆየውን የቅኝ ግዛት ዘመን ጀመሩ።
model:      ስፔናውያን ለሶስት መቶ አመታት የቆየውን የቅኝ ግዛት ዘመን ጀመሩ
```
Three "errors" in this sentence are spelling variants (ሦ/ሶ, ዓ/አ) and a full stop (።). A clinician would accept this output as correct.

**PAUSE.** What should the number be, and how do you make sure the next team scores the same way?

**Our fix.** A versioned normaliser (`src/asr/normalize.py`, `am-norm-v1`): fold homophone families (ሀ/ሐ/ኀ/ኸ, ሰ/ሠ, አ/ዐ, ጸ/ፀ), remove Ethiopic punctuation (። ፣ ፤), convert Ge'ez numerals (፳፻፲፰ -> 2018), drop `<noise>` tags, lower-case Latin. It has unit tests (`tests/test_normalize.py`), and its version is logged with every result and returned by the API. Report **CER** next to WER: Amharic words are long and carry affixes, so one wrong suffix costs a whole word in WER.

**Check.** Normalised WER **35.6%**, CER **12.4%** (raw CER 15.6%). In MLflow, every run carries `normalizer = am-norm-v1`. Open a **second Git Bash window** (`cd` to the project, `source scripts/env.sh`), because this command keeps running until you press Ctrl+C:
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000     # open http://127.0.0.1:5000, "Model training" mode
```
**Aside (2 min): is every output complete?** The `cut` column counts outputs that used every decoder token they were allowed without finishing the sentence. The first version of this lab decoded with `max_new_tokens=225`, a common default. Re-run it that way (clean channel only, about 30 s):
```bash
python -m asr.evaluate --model champion --channel clean --max-new-tokens 225
```
```
tiny-600hr-t225: WER 0.405 (raw 0.523)  CER 0.187 (raw 0.218)  size 151.1 MB  RTF 0.0528
  clean          WER 0.405  CER 0.187  empty 0.00  cut 11/40
  clean/long     WER 0.422  CER 0.217  empty 0.00  cut 11/21
```
11 long sentences stop mid-way, and CER rises from 12.4% to 18.7% with the same model and the same scorer. Amharic costs about 2.4 Whisper tokens per character (each Ge'ez letter is 3 bytes for the byte-level tokenizer); the longest reference needs 397 tokens. Whisper's decoder holds 448 positions, 4 of them taken by the prompt, so the lab now uses 440, set in `configs/params.yaml` (`decode:`) and logged with every run. A cheap check for any sequence model: count outputs that hit the limit, or compare hypothesis length with reference length.

**Lesson.** A 14-point WER difference came from scoring, not from the model, and another 5 points from one decoding setting. Never compare two numbers scored with different normalisers. **F12** wrong metric / normalisation. Rule **REP-02** (tracked runs record how they were scored).

---

## Episode 2. The split that leaks (10 min)

**Setup.** You want to fine-tune on part of these 223 clips and test on the rest.

**Wrong path.** Random 80/20 split by clip.

```bash
python scripts/split_leakage.py
```

**The issue appears.**
```
random split by clip:      178 train / 45 test, test clips whose sentence is also in train: 64%
```
FLEURS has each sentence read by up to three people. With a random split, **64%** of test clips have their exact sentence in training. A fine-tuned model will look much better than it is.

**PAUSE.** What should you group by? What would you group by with real clinic recordings?

**Our fix.** Split by group (here: sentence). With your own data, group by speaker, and also by patient, facility and day where they exist.

**Check.**
```
grouped split by sentence: 176 train / 47 test, test clips whose sentence is also in train: 0%
gender column values: {'FEMALE': 223}  <- check before you trust it
```
FLEURS publishes no speaker id, and its gender column says FEMALE for every clip (you can hear that it is wrong). You cannot report per-gender results from this metadata, and you cannot guarantee speaker-disjoint splits. Write that in the dataset card (`docs/dataset_card.md`) instead of pretending.
**F11** leakage (speaker / sentence). Rule **DATA-02** (dataset card states known issues).

---

## Episode 3. Silence that speaks (15 min)

**Setup.** In the clinic, people press Speak and say nothing, or record a fan, or upload a broken file.

**Wrong path.** Send whatever arrives straight to the model.

```bash
python scripts/silence_probe.py
```

**The issue appears.**
```
digital silence (5 s)    model says: 'አስተላለሁ'   gate: REJECT silent or near-silent (loudest frames -180.0 dBFS)
room noise (5 s)         model says: 'አስተላለሁ'   gate: PASS
fan hum 100 Hz (5 s)     model says: 'አስተላለሁ'   gate: PASS
```
The model returns a confident Amharic word for silence. Nothing crashes; the EMR would save it.

**PAUSE.** What would you check before the model runs? Which checks reject, which only warn?

**Our fix.** An audio gate (`src/asr/audio.py`), the Great Expectations idea for audio: readable file, length 0.5-30 s, not silent (loudest 5% of 25 ms frames above -50 dBFS), clipping, sample rate, channels. Reject with a reason the app can show ("no speech detected, please try again"); warn and still transcribe for 8 kHz or stereo. Run it on every request, and on the eval set before every evaluation:
```bash
python scripts/check_data.py
```
**Check.** `pytest tests/test_audio_gate.py -q` (7 passed), the data gate above (`eval clips: 40  median SNR 42.7 dB  ->  PASS`), and the gate's verdict on each sample file:
```bash
python scripts/check_data.py --samples
```
```
clipped.wav      WARN   clipping (4.9% of samples)
corrupt.wav      REJECT unreadable audio (LibsndfileError)
good.wav         PASS
not_audio.wav    REJECT unreadable audio (LibsndfileError)
phone_8k.wav     WARN   low sample rate (8000 Hz): telephone-band audio
silence.wav      REJECT silent or near-silent (loudest frames -59.5 dBFS)
stereo_44k.wav   WARN   2 channels, downmixed to mono
too_short.wav    REJECT too short (0.2 s); silent or near-silent (loudest frames -68.2 dBFS)
```

**What the gate does not catch.** Room noise and a steady hum still pass, and the model still "hears" a word. A gate reduces the problem; it does not remove it. Episode 6 watches for it in production.

A calibration lesson from building this: our first silence rule (average level below -45 dBFS) rejected **18 of 60** requests carrying real FLEURS speech in a load test, because some readers are quiet. Calibrate gates on real data, and watch the rejection rate.
**F17** silent failure, **F01** broken input. Rule **DATA-03** (validation runs before evaluation and serving).

---

## Episode 4. Better on average, worse for some (20 min)

**Setup.** A newer training stage of the same model (`tiny-stage2`) is available. Ops also asks for a smaller int8 version to save memory.

```bash
python -m asr.evaluate --model candidate
python -m asr.evaluate --model champion --quantize
```

**Wrong path.** Ship the model with the best overall WER.

**The issue appears.**
| | WER | CER | CER phone/long | size | RTF |
|---|---|---|---|---|---|
| champion `tiny-600hr` | 35.6% | 12.4% | 86.9% | 151.1 MB | 0.066 |
| candidate `tiny-stage2` | **35.4%** | **12.3%** | 95.0% | 151.1 MB | 0.066 |
| champion int8 | 51.9% | 20.7% | 103.4% | 121.5 MB | 0.069 |

The candidate wins overall by 0.2 WER points: about one word in 7 minutes of audio, well inside noise. On long, phone-quality recordings it is 8.1 CER points worse. The int8 model is 20% smaller, no faster on this CPU, and 8.3 CER points worse. (A CER above 100% is possible: on bad audio the model repeats itself, and every extra character counts as an error. The `cut` column shows how often it ran to the 440-token limit.)

**PAUSE.** Which would you deploy? What should the rule be, and who signs it, and when is the rule written?

**Our fix.** A promotion rule written **before** seeing the numbers (`configs/params.yaml`, `promotion:`): no worse overall, no group more than 2 CER points worse. A script applies it; a person signs Gate 2 (`docs/approvals/TEMPLATE_gate2.md`). CI runs the same script, so a regression fails the build.
```bash
python scripts/promote.py
python scripts/promote.py --candidate tiny-600hr-int8
```
**Check.**
```
Decision: REJECT
  - group phone CER worse by +0.021
  - group phone/long CER worse by +0.081
```
int8: REJECT, every group worse. Discuss: 40 clips cannot tell 0.2 points apart. What sample size would you need before the next decision?
**F14** silent regression after update. Rule **GOV-01** (Gate 2 record before deployment).

---

## Episode 5. Thirty clinicians press Speak (15 min)

**Setup.** Start the "production server" on your laptop: champion, candidate, a gateway (the only URL the EMR knows), a monitor and a watcher that raises alerts (episode 6).

```bash
bash scripts/start_server.sh
curl -s -F audio=@data/gate_samples/good.wav http://127.0.0.1:8000/transcribe
```
```
{"request_id":"645a40f3f45c","text":"በኢራን ውስጥ የአየር አደጋዎች ...","model_version":"tiny-600hr","normalizer_version":"am-norm-v1","latency_ms":3045.9,"warnings":[]}
```
Every answer says which model and normaliser produced it. One user sending five clips (7-12 s each):
```bash
python scripts/load_test.py --url http://127.0.0.1:8001 --users 1 --requests 5
```
p50 **2.1-2.7 s**, throughput 0.4-0.5 requests/s (two runs on the same laptop).

**Wrong path.** "It answers in 2 seconds, ship it." Now 30 clinicians, two requests each, with a 30 s client timeout:
```bash
python scripts/load_test.py --url http://127.0.0.1:8001 --users 30 --requests 2
```
**The issue appears.**
```
"ok": 17, "errors": 43, "error_kinds": ["ReadTimeout"], "p50_s": 14.74, "p95_s": 27.3, "throughput_rps": 0.28   (another run: ok 24, errors 36)
```
Between 36 and 43 of 60 dictations time out, depending on the run and the laptop. The ones that succeed wait 15-27 s. Nothing in the service logs an error: requests just queue.

**PAUSE.** What would you change: model, code, server, or the contract with the app?

**Our fix.** (1) Backpressure: when 4 requests are already waiting, answer **503 busy, retry in 3 s** at once, so the app can tell the clinician and fall back to typing. (2) Capacity from measurement: one CPU replica handles about 0.4-0.5 requests/s; size replicas for peak dictations per second, not for users. (3) Make latency a promotion criterion: `configs/params.yaml` already reserves `max_p95_latency_s: 3.0`, but `promote.py` does not check it yet, because the offline evaluation measures batch speed (RTF), not per-request latency. Who should own that number?
```bash
bash scripts/start_server.sh stop
ASR_MAX_WAITING=4 bash scripts/start_server.sh
python scripts/load_test.py --url http://127.0.0.1:8001 --users 30 --requests 2
```
**Check.**
```
"ok": 8, "errors": 52, "error_kinds": ["503"], "p50_s": 5.86, "p95_s": 7.5, "wall_s": 14.8   (another run: p95 4.8 s, wall 9 s)
```
The same laptop cannot serve 30 simultaneous users, but now it fails fast and honestly (9-15 s instead of 60 s, no hung clients). Real capacity comes from more replicas on the real server: same compose file, more CPUs.
**F16** latency / load failure. Rule **TST-02** (CI and load checks before release).

---

## Episode 6. New phones, then an outbreak: drift and the automated response (25 min)

**Setup.** The server from episode 5 is still running, including the **watcher** (port 8060). Open two browser tabs: http://127.0.0.1:8060 (the alert inbox, a stand-in for the team's Telegram group) and http://127.0.0.1:8050 (the monitor). We simulate four clinic days:

- **Day 1:** the usual headsets.
- **Days 2-3:** the facility gets a batch of cheap phones (we simulate the channel: 8 kHz, band-pass, noise).
- **Day 4:** headsets are back, but a cholera outbreak has started. Notes now contain terms the model never learned (ኮሌራ, ኦአርኤስ, አጣዳፊ የውሃ ተቅማጥ). This part is simulated at text level: the terms go into the note the clinician saves.

There are no reference transcripts in production. Clinicians do correct the text before saving, and we keep only *how much* they changed (the edit rate), never the text.

```bash
python scripts/simulate_clinic.py --url http://127.0.0.1:8000     # 4 days x 20 dictations, about 2 min
```

**Wrong path 1.** Watch only uptime, errors and latency.

**The issue appears.** Every request was answered and nothing crashed. But:
```
day 1 (usual headset): ok 20, rejected 0, empty 0, mean clinician edit rate 0.115
day 2 (new phones): ok 18, rejected 2, empty 0, mean clinician edit rate 0.792
day 3 (new phones): ok 18, rejected 2, empty 0, mean clinician edit rate 0.792
day 4 (outbreak notes): ok 20, rejected 0, empty 0, mean clinician edit rate 0.378
```
On the new phones, clinicians rewrite four fifths of every note. In the offline evaluation (episode 4) the phone channel had CER 83%: the model is unusable on these devices. On day 4 the audio is fine again, yet clinicians still correct three times more than on day 1.

**Wrong path 2.** The first fix most teams propose: "send an alert whenever a request looks bad". Replay the same four days through that rule:
```bash
python -m asr.watcher --replay --mode naive
```
```
naive mode: 82 messages would be sent {'edits': 52, 'snr': 26, 'rejected': 4}
```
82 messages in four days, for one facility. By day 2 the group has muted the channel, and the alert that matters is lost among the others.

**PAUSE.** (1) Which signals could have told you on day 2, without any labels? (2) How do you alert so that people keep reading the alerts? (3) What may the system do on its own, and what must a person decide? (4) Are days 2-3 and day 4 the same kind of problem?

**Our fix.** Three layers.

1. **Label-free signals** per window of requests (`src/asr/monitor.py`): median signal-to-noise ratio (a microphone fingerprint), empty outputs, rejections, latency, and the clinician edit rate.
2. **Alert rules written down** in `configs/params.yaml` (`alerts:`), approved with the monitoring plan (`docs/MONITORING_PLAN.md`). A rule fires only when it holds for **2 windows in a row**. Each rule has a severity (P1 now, P2 same day) and a named owner. The watcher sends **one message per problem** and one when it clears.
3. **Safe automatic actions**, listed in the rule: queue the high-edit clips for annotation (by clip id, never audio), and open a **draft Gate 3 review** with the evidence. Nothing here retrains or promotes a model: that is a person's decision.

The rules also tell the two problems apart:
- **Data drift:** SNR down and edits up. The inputs changed.
- **Concept drift:** SNR normal and edits up. The right answer changed.

**Check.** The inbox at http://127.0.0.1:8060 (or `python -m asr.watcher --replay` in the terminal) shows three messages:
```
P2  Data drift: new microphones or phones (SNR dropped)
    snr_db_median = 3.4 for 2 windows in a row (threshold < 15)
    requests 21-40: SNR median 5.0, 3.4 dB; edit rate 0.6, 0.984
    Owner: ASR lead. Next step: find the device batch, evaluate the champion on its audio, Gate 3 review
    ✓ 18 clips to the annotation queue (data/annotation/queue.csv)
    ✓ draft Gate 3 record logs/gate3/gate3_DRAFT_<date>_channel_shift.md
RESOLVED  new microphones or phones (SNR dropped): back to normal
    window 61-70: snr_db_median = 43.5
P2  Concept drift: clinicians correct more, audio unchanged
    edit_rate = 0.415 for 2 windows in a row (threshold > 0.3)
    requests 61-80: SNR median 43.5, 40.7 dB; edit rate 0.342, 0.415
    Owner: ASR lead. Next step: read the corrected notes for new terms, plan a retraining (new candidate, then Gate 2)
    ✓ 20 clips to the annotation queue (data/annotation/queue.csv)
    ✓ draft Gate 3 record logs/gate3/gate3_DRAFT_<date>_notes_changed.md
```
Three messages instead of 82. The data-drift alert fires at the end of day 2, from audio statistics alone, before any clinician complains. Persistence costs one window of delay; that is the price of not crying wolf. Open a Gate 3 draft (`cat logs/gate3/*channel_shift.md`): the evidence table is already filled in, and the decision field says **pending, a person decides**. The monitor page (http://127.0.0.1:8050, or `python -m asr.monitor --window 20`) shows the same story per window:
```
model_version window  requests  rejected  empty_rate  p50_ms   p95_ms  snr_db_median  edit_rate        alerts
   tiny-600hr   1-20        20       0.0         0.0 1400.50 2023.920           42.4      0.115             -
   tiny-600hr  21-40        20       0.1         0.0 1227.60 3015.415            4.8      0.792 CHANNEL,EDITS
   tiny-600hr  41-60        20       0.1         0.0 1225.45 2988.140            4.8      0.792 CHANNEL,EDITS
   tiny-600hr  61-80        20       0.0         0.0 1430.80 2068.350           42.4      0.378         EDITS
```
To send the same messages to a real Telegram group, put `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` in `.env` (never in git). Messages that cannot be sent (no internet) are kept and retried. The queue feeds the next training round (episodes 8 and 9).
**F08** acquisition / device drift, **F05** concept drift, **F20** feedback loop missing. Rule **MON-01** (`docs/MONITORING_PLAN.md`).

---

## Episode 7. Canary, rollback, and what the logs know (15 min)

**Setup.** Ops insists on trying the int8 model in production "because it is smaller". Restart with it as the candidate, and send it half the traffic:

```bash
bash scripts/start_server.sh stop
CANDIDATE_PATH=models/tiny-600hr CANDIDATE_VERSION=tiny-600hr-int8 CANDIDATE_QUANTIZE=1 ASR_MAX_WAITING=4 bash scripts/start_server.sh
curl -s -X POST -F percent=50 http://127.0.0.1:8000/admin/canary
python scripts/simulate_clinic.py --url http://127.0.0.1:8000 --days 1 --per-day 40
python -m asr.monitor --window 40
```
**The issue appears.** Watch the inbox while the clinic runs. After 30 requests:
```
P1  Canary: candidate clearly worse than champion
    edit_rate_gap = 0.068 for 2 windows in a row (threshold > 0.05)
    requests 1-30: SNR median 42.5, 42.1 dB; edit rate 0.147, 0.2
    tiny-600hr-int8 vs champion edit rate gap: 0.069, 0.068
    Owner: Ops owner. Next step: keep the champion; the candidate goes back to validation
    ✓ canary rolled back: gateway now sends 0% to the candidate
```
Same audio quality, but clinicians edit the int8 output much more. The *absolute* edit rate never crossed 0.30, so the episode 6 rules alone would have stayed silent. Here the comparison is **between versions on the same traffic**: each check pools 2 windows (about 10 corrected notes per version), because 5 notes are too few to judge.

**PAUSE.** The watcher rolled back on its own. Should it have? What would you never let it do on its own?

**Our fix, and why the rollback may be automatic.** Rollback is safe and reversible: the champion has already passed Gate 2, and one call restores it. The rule and its action were written in the Gate 2 record before the canary started. Promotion is the opposite: it is never automatic. Check that the gateway is back on the champion; the manual rollback is the same single call:
```bash
curl -s http://127.0.0.1:8000/health | grep -o '"canary_percent":[0-9.]*'
#  "canary_percent":0.0
curl -s -X POST -F percent=0 http://127.0.0.1:8000/admin/canary          # manual rollback, one call
python -m asr.monitor --window 40
```
The monitor shows how many requests reached the int8 model before the rollback (about 15 of 40; the exact number depends on timing). The gateway splits by a hash of the recording, so the same clip always goes to the same model. In production, hash the facility or clinician id.

**Privacy, same episode.** A developer turns on transcript logging "to debug":
```bash
mkdir -p logs/naive
ASR_LOG_TEXT=1 ASR_LOG_DIR=logs/naive python -m uvicorn asr.service:app --port 8003 > logs/naive.out 2>&1 &
until curl -s http://127.0.0.1:8003/health | grep -q ready; do sleep 2; done     # about 10 s
curl -s -o /dev/null -F audio=@data/gate_samples/good.wav http://127.0.0.1:8003/transcribe
tail -1 logs/naive/requests.jsonl
```
The log line now ends with `"text": "..."`: in a clinic that is a diagnosis, a name, a phone number, kept for 90 days and copied to every log tool. The default log has only lengths, levels, latency and versions, and `tests/test_service.py::test_logs_never_contain_transcript` fails the build if text ever appears.
**F19** unsafe deployment (rollback), **F18** privacy leak in logs. Rules **GOV-01**, **SEC-02** (PDPP 1321/2024), **LOG-01**.

```bash
bash scripts/start_server.sh stop          # stops every service, including the debug one on :8003
rm -rf logs/naive                          # delete the transcripts you just logged
```

---

## Wrap-up. Does this project meet the framework? (5 min)

Run the workshop toolkit on the project you just used. Install it once from the take-home pack (`toolkit/` on the USB stick):
```bash
uv pip install path/to/eaii_mlops-0.1.0-py3-none-any.whl --python .venv/Scripts/python.exe
eaii-mlops check .
```
```
[ERROR] GOV-01  Gate 2 approval record exists before deployment (docs/approvals/)
[WARN] LOG-01  Structured logging in src/ (no bare print), no personal data logged
Summary: 1 rule(s) with errors (1 findings), 1 rule(s) with warnings (8 findings), 16 passed.
```
We have been serving the champion all morning without a signed Gate 2 record. Write it now, as the approver would, with the evidence from episodes 1-7:
```bash
eaii-mlops gate --stage 2 --model tiny-600hr --version 1 --approver "<your name>" --role "ASR lead" \
  --decision approved-with-conditions --conditions "Pilot facilities, headsets only; watcher alerts on" \
  --evidence model_card=docs/model_card.md dataset_card=docs/dataset_card.md evaluation=results/tiny-600hr.json \
             monitoring_plan=docs/MONITORING_PLAN.md rollback=tiny-600hr --no-input
eaii-mlops check .
```
```
Summary: 0 rule(s) with errors (0 findings), 1 rule(s) with warnings (8 findings), 17 passed.
```
Open the record in `docs/approvals/`: the evidence still marked *pending* (MLflow run link, fairness results) is what a real approver would ask for before signing. The LOG-01 warning is the command-line reports printing to the terminal, an exception the README explains. A green check is the floor, not the ceiling: it proves the evidence exists, not that the model is good.

---

## Episode 8 (optional). Do our annotators agree? (15 min)

The annotation queue from episode 6 needs transcripts. Two annotators transcribe the same 20 clips.
```bash
python scripts/annotation_agreement.py
```
```
clips: 20
disagreement (CER between annotators)  raw: 0.045   normalised: 0.019
clips that look different (raw CER > 0.02): 15   really different (normalised CER > 0.02): 7
  G1 numbers as words in verbatim layer: 4 transcript(s) by annotator B break it
  G2 no untagged personal names (use <name>): 1 transcript(s) by annotator B break it
  G3 Latin in lower case: 1 transcript(s) by annotator B break it
-> 7 clips to adjudication: data/annotation/adjudication.csv
```
**PAUSE.** Which differences matter? Which should the guideline settle once instead of the annotators arguing per clip?
Normalise first (homophones and punctuation are not disagreements), then adjudicate only real differences, and check guideline rules automatically. Label Studio does this at scale (two annotators per task, agreement, review queue); it is optional here. **F10** biased / inconsistent labels.

## Episode 9 (take-home). Fine-tune in Colab

`notebooks/finetune_colab.ipynb` (**not tested in Colab**): 400 steps on one pinned shard of open WAXAL Amharic train data (CC-BY-SA-4.0, 8.9 h, 186 speakers), with a SHA-256 check, a speaker-overlap check against WAXAL test, and MLflow logging. Then bring the model back and run `scripts/promote.py`. Did your model pass the rule? (The facilitators' 35 h run did better on WAXAL and worse on FLEURS: see `docs/AMHARIC_SPEECH_DATA.md`.)

---

## Where this ends up

The same `/transcribe` contract is used by the EthioEMR assistant's Speak button (OpenMRS module). The production model is a larger Amharic model (Ethio-ASR, CC-BY-4.0, about 19% WER / 6% CER on FLEURS), served on a server, behind the same gate, logging and monitoring. Moving our lab stack to that server changes one URL.

## If something fails

Pre-computed outputs are in `results/precomputed/` (evaluation JSON and hypotheses for all three models, load-test results, request, feedback and alert logs, monitor page). `python scripts/promote.py` falls back to them when `results/` has no run of its own; the monitor page from episode 6 is `results/precomputed/monitor_after_clinic.html`.
