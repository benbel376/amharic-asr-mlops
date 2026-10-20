"""Unpack the FLEURS Amharic dev split and build the lab datasets.

    python scripts/prepare_data.py

Writes (all under data/, ignored by git):
  clips/<id>.wav            223 original recordings (16 kHz mono)
  manifest.csv              one row per clip: clip_id, sentence_id, text, gender, duration_s, n_words
  eval.csv                  the fixed evaluation subset (configs/params.yaml: eval.n_clips)
  shift/<id>.wav            the same eval clips through a simulated phone/cheap-mic channel
  gate_samples/             files that the audio gate must reject or flag
"""
from __future__ import annotations

import csv
import sys
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import yaml
from scipy.signal import butter, resample_poly, sosfilt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
P = yaml.safe_load((ROOT / "configs" / "params.yaml").read_text(encoding="utf-8"))


def unpack() -> None:
    clips = DATA / "clips"
    if clips.exists() and len(list(clips.glob("*.wav"))) >= 223:
        return
    clips.mkdir(parents=True, exist_ok=True)
    with tarfile.open(DATA / "raw" / "dev.tar.gz") as tar:
        for m in tar.getmembers():
            if m.isfile() and m.name.endswith(".wav"):
                m.name = Path(m.name).name  # flatten, and never trust paths inside archives
                tar.extract(m, clips, filter="data")


def manifest() -> pd.DataFrame:
    cols = ["sentence_id", "file", "raw_text", "norm_text", "chars", "num_samples", "gender"]
    df = pd.read_csv(DATA / "raw" / "dev.tsv", sep="\t", names=cols, quoting=csv.QUOTE_NONE)
    df["clip_id"] = df["file"].str.replace(".wav", "", regex=False)
    df["duration_s"] = (df["num_samples"] / 16000).round(2)
    df["text"] = df["raw_text"]
    df["n_words"] = df["text"].str.split().str.len()
    out = df[["clip_id", "sentence_id", "text", "gender", "duration_s", "n_words"]]
    out.to_csv(DATA / "manifest.csv", index=False, encoding="utf-8")
    return out


def eval_subset(df: pd.DataFrame) -> pd.DataFrame:
    """Fixed, reproducible subset: one clip per sentence, seeded sample."""
    one = df.drop_duplicates("sentence_id")
    ev = one.sample(n=P["eval"]["n_clips"], random_state=P["seed"]).sort_values("clip_id")
    ev.to_csv(DATA / "eval.csv", index=False, encoding="utf-8")
    return ev


def phone_channel(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Simulate a cheap phone / headset: 8 kHz, 300-3400 Hz band, noise, louder gain."""
    y = resample_poly(x, 1, 2)                       # 16 kHz -> 8 kHz
    sos = butter(4, [300, 3400], btype="band", fs=8000, output="sos")
    y = sosfilt(sos, y)
    y = y + rng.normal(0, P["shift"]["noise_std"], len(y))
    y = np.clip(y * P["shift"]["gain"], -1, 1)
    return y.astype("float32")


def shift_set(ev: pd.DataFrame) -> None:
    out = DATA / "shift"
    out.mkdir(exist_ok=True)
    rng = np.random.default_rng(P["seed"])
    for cid in ev["clip_id"]:
        x, sr = sf.read(DATA / "clips" / f"{cid}.wav", dtype="float32")
        sf.write(out / f"{cid}.wav", phone_channel(x, rng), 8000)


def gate_samples(ev: pd.DataFrame) -> None:
    out = DATA / "gate_samples"
    out.mkdir(exist_ok=True)
    sr = 16000
    rng = np.random.default_rng(P["seed"])
    x, _ = sf.read(DATA / "clips" / f"{ev['clip_id'].iloc[0]}.wav", dtype="float32")
    sf.write(out / "good.wav", x, sr)
    sf.write(out / "silence.wav", rng.normal(0, 0.001, sr * 5).astype("float32"), sr)
    sf.write(out / "too_short.wav", x[: sr // 5], sr)
    sf.write(out / "clipped.wav", np.clip(x / (np.abs(x).max() + 1e-9) * 3, -1, 1), sr)
    sf.write(out / "stereo_44k.wav", np.stack([resample_poly(x, 441, 160)] * 2, axis=1), 44100)
    sf.write(out / "phone_8k.wav", resample_poly(x, 1, 2), 8000)
    (out / "corrupt.wav").write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt " + bytes(20))
    (out / "not_audio.wav").write_text("this is a text file renamed to .wav", encoding="utf-8")


def main() -> int:
    unpack()
    df = manifest()
    ev = eval_subset(df)
    shift_set(ev)
    gate_samples(ev)
    print(f"clips: {len(df)}  sentences: {df.sentence_id.nunique()}  "
          f"audio: {df.duration_s.sum() / 60:.1f} min  gender values: {df.gender.value_counts().to_dict()}")
    print(f"eval subset: {len(ev)} clips, {ev.duration_s.sum() / 60:.1f} min  ->  data/eval.csv")
    print(f"shifted copies: {len(ev)} -> data/shift/   gate samples -> data/gate_samples/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
