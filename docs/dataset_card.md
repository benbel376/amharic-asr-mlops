# Dataset card: lab evaluation set (FLEURS Amharic dev subset)

- **Source:** google/fleurs `am_et` dev split, CC-BY-4.0. See `data/SOURCES.md`.
- **Content:** 223 recordings of 124 read Wikipedia sentences, 16 kHz mono, 39.2 min total.
- **Eval subset (`data/eval.csv`):** 40 clips, one per sentence, `seed: 0`, 7.1 min. Fixed: do not change it without a new version name, or results stop being comparable.
- **Shifted copy (`data/shift/`):** the same 40 clips through a simulated cheap phone channel (8 kHz, 300-3400 Hz band-pass, white noise std 0.01, gain 2). Median SNR falls from 42.7 dB to 4.7 dB.
- **Known issues:** speaker ids are not published; the `gender` column is constant (`FEMALE`); the same sentence appears up to 3 times (split by sentence, episode 2); no clinical vocabulary; read speech, not dictation; possible overlap with the models' training data.
- **Personal data:** none beyond the voices of consenting FLEURS readers. Do not add patient audio to this folder (SEC-02).
- **Owner:** ASR facilitators (Abenezer, Biniyam). Review before each workshop run.
