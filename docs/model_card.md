# Model card: Amharic ASR (lab champion `tiny-600hr`, candidate `tiny-stage2`)

| | champion `tiny-600hr` | candidate `tiny-stage2` | champion int8 |
|---|---|---|---|
| Base | Whisper tiny (39 M params) | same | same, dynamic int8 Linear layers |
| Licence | Apache-2.0 | Apache-2.0 | Apache-2.0 |
| Size (state dict) | 151.1 MB | 151.1 MB | 121.5 MB |
| WER / CER, clean, `am-norm-v1` | 35.6% / 12.4% | 35.4% / 12.3% | 51.9% / 20.7% |
| WER / CER, raw text | 49.3% / 15.6% | 49.3% / 15.6% | 62.4% / 23.5% |
| CER, phone channel | 82.8% | 84.9% | 113.0% |
| CPU real-time factor (batch 8) | 0.066 | 0.066 | 0.069 |

Evaluated 20 Oct 2026 on the 40-clip FLEURS am dev subset (`data/eval.csv`), Windows 11, 24 logical CPUs, 12 torch threads, greedy decoding, `max_new_tokens=440` (`configs/params.yaml`). Earlier versions of this card used 225 tokens, which cut 11 of 40 outputs (WER 40.5% / CER 18.7% for the champion).

- **Intended use:** teaching. Short Amharic dictation, one speaker, a good headset.
- **Not for:** clinical use; code-switched English medical terms (not tested; a larger internal benchmark found no off-the-shelf model writes them correctly); phone-quality audio (CER above 80%; on noise the model repeats itself until the token limit).
- **Known failure:** on silence or steady noise the model outputs a fixed phrase (`አስተላለሁ`) instead of nothing. The audio gate rejects silence; it does not catch a steady hum.
- **Training data:** not stated by the model author. FLEURS overlap is possible.
- **Production pick (outside the lab):** Ethio-ASR (`badrex/Ethio-ASR-amharic`, CC-BY-4.0), 19.4% WER / 5.8% CER on FLEURS test, n = 200, from an internal benchmark (not re-run here).
