# Open Amharic speech data: what we can train and evaluate on

Survey done on 20 Oct 2026. Every licence, size and metadata field below was read on the dataset's own page (Hugging Face card or API, Mozilla Data Collective datasheet, OpenSLR, LDC catalogue). "Unverified" means we could not confirm it there. Hours for FLEURS are summed from its TSVs.

## Datasets

| Dataset | Licence | Size | Speakers | Metadata | Access | Use it for |
|---|---|---|---|---|---|---|
| **WAXAL** `google/WaxalNLP`, `ASR/amh` | **CC-BY-SA-4.0** (Digital Umuganda languages, per card table) | train 38,022 utt / 189.7 h (21 parquet shards, 10.5 GB); validation 2,912 / 14.0 h; test 3,420 / 16.2 h (2 shards, 0.92 GB); 80k unlabelled | 566 train / 28 val / 24 test, **speaker-disjoint** | `speaker_id`, `gender` (Male/Female); no age or accent | HF parquet, not gated; first HF commit 19 Jan 2026 | **Training (recommended)**, and per-gender evaluation on test |
| **FLEURS** `google/fleurs` `am_et` | CC-BY-4.0 | train 3,163 utt / 11.1 h (1.95 GB); dev 223 / 0.65 h; test 516 / 1.61 h | no speaker id | `gender` only; **dev and test are 100% "FEMALE"** (wrong; train is 2,240 M / 923 F) | HF TSV + tar.gz, not gated | Lab's current eval (read speech). Not for per-gender results. Likely in most models' training data |
| **Horn-ASR** `LesanAI/Horn-ASR`, config `am` | CC-BY-SA-4.0 | test only: 1,000 utt / 4.38 h, 16 kHz WAV files | 524 (interview proxy) | `gender` (annotator-checked, 518 M / 482 F), `dialect` (Addis, Gojam, Gonder, Shewa, Wollo, L2), `domain` (incl. Health) | HF, one WAV per clip, not gated | **Held-out evaluation with gender and dialect**. Spontaneous interviews; card says "public-domain interview material" (provenance unverified) |
| Dataset.ET `snapwre/amharic-speech` v0.2.0 | CC-BY-4.0 | 16,866 clips / 51.5 h (5.8 GB); train 42.1 h, val 4.7 h, test 4.7 h | 493, speaker- and prompt-disjoint | speaker id, gender (69% M), age band, region | HF parquet, not gated | Training (read speech via Telegram, Opus codec), second per-gender test |
| Afrivoice_Ethiopia `DigitalUmuganda/Afrivoice_Ethiopia` | CC-BY-4.0 on this card (WAXAL lists the same source as CC-BY-SA) | Amharic 618 h, 219.5 h transcribed; repo 213 GB (5 languages) | not stated | creator id, gender, age group, location | HF, **gated** (accept conditions) | Larger training pool; WAXAL Amharic is derived from it, so deduplicate against WAXAL test |
| ALFFA, OpenSLR SLR25 | MIT | about 20 h train + 2 h test, 1.0 GB | not stated | Kaldi files, no gender | tarball, no sign-up | Training (read news). Only Amharic set on OpenSLR |
| Common Voice Scripted 27.0, Amharic (Mozilla Data Collective) | CC0-1.0, plus MDC terms: "It is forbidden to re-host or re-share this dataset" | 1,728 clips; 1.94 h validated; 62 MB | 56 | age, gender (0 female clips declared), accent | MDC download with terms | Too small; cannot be put on our USB stick |
| Common Voice Spontaneous 5.0, Amharic | CC0-1.0, same MDC terms | 1 clip | 1 | | MDC | Unusable |
| leyu-amharic dialect sets (e.g. `leyu-amharic-gonder-dialect`) | CC-BY-4.0 (Gojjam set: "other") | Gonder: 112.8 h, 67 speakers | | speaker id, gender, dialect | HF, not gated | Dialect evaluation / training (only the Gonder card was read) |
| YODAS2 `espnet/yodas2` `am000` | CC-BY-3.0 (videos "only CC-licensed") | ~76 h, 10.9 GB, YouTube subtitles | | none | HF | **Licence and label risk**: per-video licence and subtitle quality unverifiable. Avoid |
| VoxLingua107 `am` | CC-BY-4.0 | 81 h YouTube | | language id only, **no transcripts** | zip | Language ID / self-supervised only |
| BibleTTS; Omnilingual ASR Corpus | CC-BY-SA; CC-BY-4.0 | | | | | **No Amharic** in either (checked) |
| IARPA Babel Amharic, LDC2019S22 | LDC licence, fee | ~204 h telephone speech | | dialect region | LDC membership/fee | **Not open**. Not for the lab |
| AfriSwitch / AfriSwitchCare (`intronhealth`) | CC-BY-NC-SA-4.0 | ~5 h code-switched | | | gated, manual approval | Non-commercial, evaluation only |

## Contamination: which test is clean for which model

| Model | Training data (from its card) | FLEURS-40 (lab) | WAXAL test | Horn-ASR test |
|---|---|---|---|---|
| `b1n1yam/shook-tiny-amharic-600hr`, `-stage2-polish` (current lab models) | **not stated** ("None dataset"); uploaded 18-21 Nov 2025 | possibly seen | **clean**: WAXAL was first published on HF on 19 Jan 2026, two months after these models | probably clean (no public copy before 2026; unverified) |
| `waxal-benchmarking/whisper-tiny-waxal-amh` | WAXAL Amharic train | probably clean (card names only WAXAL) | in-domain, speaker-disjoint | probably clean |
| `badrex/Ethio-ASR-amharic` | WAXAL Amharic train (card); FLEURS per an internal review | seen | in-domain, speaker-disjoint | probably clean |
| Our fine-tune `tiny-600hr-waxal` | `tiny-600hr` + 4 WAXAL train shards | inherits `tiny-600hr`'s risk | in-domain, speaker-disjoint | probably clean |

Plan: (1) keep FLEURS-40 as the lab's fixed regression set, labelled "may be seen"; (2) report WAXAL test (speaker-disjoint, gender labels) and Horn-ASR (no known model trained on it, gender and dialect labels) beside it; (3) never pick training data or tune on Horn-ASR or WAXAL test; (4) when someone hands you a model without a data statement, test it on data published after the model was.

## Model check (20 Oct 2026)

All rows were scored with the lab's own `asr.metrics` and `am-norm-v1` (normalised WER / CER), greedy decoding, by `_gpu/eval_models.py`. Its 225-token row reproduces the lab's old setting exactly (40.5 / 18.7), and its 440-token row reproduces today's `python -m asr.evaluate --model champion` exactly (35.6 / 12.4; same 40 hypotheses). Eval sets:
- **FLEURS-40**: the lab's `data/eval.csv`, 7.1 min.
- **WAXAL-80**: test shard 0 (SHA-256 `6d6e9fad…`, matches the HF LFS id), 40 female + 40 male clips, 22 speakers, 23.3 min.
- **Horn-80**: 40 female + 40 male clips, 80 different speakers, 2-28 s each, 15.9 min.

CPU RTF is FLEURS-40 at batch 8 with 12 torch threads on a 24-thread laptop; WER/CER were computed on the GPU, and CPU and GPU gave the same WER and CER.

| Model | Licence | Size | CPU RTF | FLEURS-40 | WAXAL-80 (F / M WER) | Horn-80 (F / M WER) |
|---|---|---|---|---|---|---|
| `tiny-600hr`, lab champion, `max_new_tokens=225` (lab before 20 Oct) | Apache-2.0 | 151 MB | 0.048 | 40.5 / 18.7 | 75.7 / 50.9 | 71.3 / 46.9 |
| `tiny-600hr`, `max_new_tokens=440` (lab today) | Apache-2.0 | 151 MB | 0.052 | **35.6 / 12.4** | 68.5 / 32.3 (74.1 / 63.0) | 65.5 / 35.7 (65.0 / 66.1) |
| `tiny-stage2`, 440 tokens | Apache-2.0 | 151 MB | 0.058 | 35.4 / 12.3 | 68.3 / 32.8 (74.1 / 62.5) | **62.6 / 32.3** (62.9 / 62.2) |
| `waxal-benchmarking/whisper-tiny-waxal-amh` @`8eb9f14`, 440 | CC-BY-4.0, data = WAXAL | 151 MB | 0.063 | 40.2 / 16.0 | **41.6 / 19.5** (44.6 / 38.6) * | 68.5 / 41.6 (67.1 / 70.0) |
| `b1n1yam/shook-900-base` @`62186a1` (Whisper base), 440 | **none on card** | 290 MB | 0.117 | 43.5 / 14.2 | 72.1 / 31.4 (76.4 / 67.9) | 71.6 / 37.1 (70.6 / 72.6) |
| **ours** `tiny-600hr-waxal` (`tiny-600hr` + 35 h WAXAL, 600 steps), 440 | Apache-2.0 base + CC-BY-SA data | 151 MB | 0.059 | 38.9 / 13.9 | 57.8 / 26.4 (60.4 / 55.2) * | 68.2 / 37.0 (72.5 / 63.6) |
| `badrex/Ethio-ASR-amharic` (w2v-BERT CTC, 606 M), upper bound | CC-BY-4.0 | 2,424 MB | 0.126 | 14.2 / 3.9 † | 27.4 / 10.7 (28.0 / 26.8) * | **43.3 / 19.8** (43.3 / 43.4) |

\* in-domain: trained on WAXAL train (speaker-disjoint from test). † FLEURS was in Ethio-ASR's training data.

Not run: `openai/whisper-small` and `-base` zero-shot (Whisper-small scores WER 174 on the GSMA telecom bench, and Whisper-large-v3 scores 111-153% in an internal benchmark, so they are not worth the download); Omnilingual `omniASR-CTC-300M` (Apache-2.0, 325 M; needs fairseq2, not transformers; an internal benchmark measured CTC-1B at 38.6 / 12.7 on WAXAL); MMS-1b-all and Ethio-ASR-multilingual 94M/300M/1B (non-commercial licences); `waxal-benchmarking/whisper-small-waxal-amh` (CC-BY-4.0, 242 M, card WER 33.6 on WAXAL; a 967 MB download that the slow link did not allow).

### Findings

1. **The lab truncated its own output (fixed 20 Oct).** `max_new_tokens=225` cut 11 of the 40 FLEURS hypotheses (10 references need more than 225 tokens; the longest needs 397) and 78 of 80 WAXAL hypotheses: Amharic needs about 2.4 byte-level tokens per character, and the median WAXAL reference is 332 tokens. With 440 tokens (Whisper's decoder allows 448, including 4 prompt tokens), the *same* champion scores **35.6 / 12.4** instead of 40.5 / 18.7. The lab now decodes with 440 (`configs/params.yaml`, `decode:`), counts outputs that hit the limit (`cut` in `asr.evaluate`), and episode 1 shows the old setting as an aside.
2. **The FLEURS score flatters the lab models.** On WAXAL, published two months after they were uploaded and therefore unseen, they score 68% WER / 32% CER. On Horn-ASR they score 63-66% / 32-36%.
3. **A per-gender gap exists and can be shown.** On WAXAL, the champion is 11 WER points worse for women (74.1 vs 63.0); every Whisper model shows the same direction. On Horn-ASR the gap disappears or flips. Two sets disagree, so this is a small-sample and speaker-mix question (8 male speakers in WAXAL-80), which makes a good discussion for episode 4.
4. **Our open-data fine-tune is not a better champion.** 35 h of WAXAL (17.9 min on the GPU) cut WAXAL WER by 10.7 points, but it cost 3.3 points on FLEURS and 2.7 on Horn-ASR. A shorter shard-0 run (400 steps, 8.9 h) shows the same pattern (39.7 / 58.6 / 67.2 WER). So it is better in-domain, worse elsewhere: the episode-4 lesson again. We therefore **left the lab champion unchanged**. The model is kept locally in `models/tiny-600hr-waxal/`, with `training_record.json` (base revision, the four shard SHA-256s, dataset revision, filters, every hyperparameter, seed 0, code and weights SHA-256) and MLflow run `613fc6e5…` in `_gpu/mlflow.db`.
5. **Only Ethio-ASR is clearly better everywhere.** It is CC-BY-4.0 and runs on this CPU at RTF 0.13. It costs a 2.4 GB download and about 3 GB of RAM, and it needs a CTC engine class. It suits a facilitator demo or a USB-stick option rather than 30 laptops on venue Wi-Fi.

## Recommendation

- **Fine-tuning data:** WAXAL Amharic train (190 h, 566 speakers, gender labels, not gated, pinned revision `5f4d8ca`). Add Dataset.ET (CC-BY-4.0, 42 h read speech) and ALFFA (MIT, 20 h) for read-speech balance. Keep FLEURS train out of any model you will score on FLEURS. The Colab notebook (`notebooks/finetune_colab.ipynb`) uses one pinned WAXAL shard with a speaker-leakage check.
- **Clean held-out evaluation with gender metadata:** WAXAL test for any model released before 19 Jan 2026 (including both lab models), plus **Horn-ASR am** (CC-BY-SA-4.0, annotator-checked gender, dialect, health domain, and no model we know of trained on it). Horn-ASR is easy for participants to fetch, because each clip is a separate 16 kHz WAV: an 80-clip subset is about 30 MB, against 0.5 GB for a WAXAL test shard.
- **Lab changes:** (a) `max_new_tokens=440`: **applied 20 Oct**, every episode re-run and every number in the guide updated; (b) a Horn-80 per-gender table in episode 4: not applied (needs a 30 MB extra download per laptop).
- **Licence notes:** WAXAL and Horn-ASR are ShareAlike. Keep attribution, and get a legal view before publishing weights trained on them. Common Voice terms forbid re-sharing, so it cannot go on the USB stick. MMS is non-commercial. `shook-900-base` has no licence on its card, so the author should add one before anyone uses it.

## Reproduce

GPU scripts live in `_gpu/` (gitignored; separate venv `_gpu/.venv` with torch 2.8.0+cu128, so the participant CPU environment is untouched):
```bash
_gpu/.venv/Scripts/python.exe -I _gpu/make_waxal_eval.py <amh-test-00000.parquet> _gpu/data/waxal_eval 40
_gpu/.venv/Scripts/python.exe _gpu/eval_models.py tiny-600hr-t440 models/tiny-600hr --mnt 440 --sets fleurs,waxal,horn
_gpu/.venv/Scripts/python.exe _gpu/finetune_waxal.py --base models/tiny-600hr --out models/tiny-600hr-waxal \
    --shards _gpu/data/waxal/amh-train-0000{0,1,2,3}.parquet --steps 600 --lr 5e-6 --warmup 50 --bs 32 --seed 0
```
The Horn-80 subset is `_gpu/data/horn/selected.csv`: seed 0, one clip per speaker, from `LesanAI/Horn-ASR` @`a2a8ab2`.
