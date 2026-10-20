# Data and model sources

All downloads are fetched by `scripts/download.py` and verified against SHA-256 checksums in `scripts/downloads.json`. Nothing here is committed to git (DATA-01).

| Item | Source | Size | Licence | Verified |
|---|---|---|---|---|
| FLEURS Amharic (`am_et`) dev split: `dev.tsv` + `audio/dev.tar.gz` | https://huggingface.co/datasets/google/fleurs (Conneau et al., 2022) | 112 MB, 223 clips, 124 sentences, 39.2 min | **CC-BY-4.0** (dataset card) | 20 Oct 2026 |
| Model `b1n1yam/shook-tiny-amharic-600hr` @ `7730909` | Hugging Face | 151 MB | **Apache-2.0** (model card) | 20 Oct 2026 |
| Model `b1n1yam/shook-tiny-amharic-stage2-polish` @ `4dc4d84` | Hugging Face | 151 MB | **Apache-2.0** (model card) | 20 Oct 2026 |

Notes:
- FLEURS is read Wikipedia sentences (FLoRes text), not clinical speech. Each sentence is read by several people, which matters for splitting (episode 2).
- FLEURS's `gender` column says `FEMALE` for all 223 dev clips (and all 516 test clips). Treat it as unreliable; we do not report gender groups from it.
- Both models are Whisper-tiny fine-tunes by one of the facilitators (Biniyam Daniel). Their training data is not stated on the model cards, so **FLEURS may be in their training data**: treat our numbers as a teaching comparison, not a benchmark.
- Attribution required by CC-BY-4.0: "FLEURS, Google, CC-BY-4.0" appears in this file and in `docs/dataset_card.md`.
- Not used (licence): MMS-1b-all and SeamlessM4T (CC-BY-NC-4.0), MMS-TTS (CC-BY-NC-4.0).
- Optional "production" model for comparison, not downloaded by default: `badrex/Ethio-ASR-amharic` (CC-BY-4.0, 2.4 GB).
- Not used by the lab, kept locally only (never published): `models/tiny-600hr-waxal`, our fine-tune of `tiny-600hr` on WAXAL Amharic train shards 0-3 (google/WaxalNLP @ `5f4d8ca`, CC-BY-SA-4.0, 35.4 h). It is better on WAXAL test, worse on FLEURS-40 and Horn-ASR, so the champion is unchanged. Training record, data hashes and results: `models/tiny-600hr-waxal/training_record.json` and `docs/AMHARIC_SPEECH_DATA.md`.
- Open Amharic speech data for training and held-out evaluation (WAXAL, Horn-ASR, Dataset.ET, ALFFA, Common Voice terms): see `docs/AMHARIC_SPEECH_DATA.md`.
