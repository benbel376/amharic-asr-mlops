"""Audio loading and the audio input gate (the Great Expectations idea, for audio)."""
from __future__ import annotations

import io
from dataclasses import dataclass, field

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

TARGET_SR = 16000

DEFAULT_GATE = {"min_sr": 16000, "min_s": 0.5, "max_s": 30.0, "silence_db": -50.0,
                "max_clip": 0.01, "min_hf_ratio": 0.0005}


@dataclass
class GateResult:
    ok: bool
    issues: list[str] = field(default_factory=list)    # reject the request
    warnings: list[str] = field(default_factory=list)  # transcribe, but flag it
    features: dict = field(default_factory=dict)       # safe to log: no content


def read_audio(data):
    """Return (float32 samples [n, channels], sample rate, channels)."""
    src = io.BytesIO(data) if isinstance(data, (bytes, bytearray)) else data
    x, sr = sf.read(src, dtype="float32", always_2d=True)
    return x, sr, x.shape[1]


def to_model_input(x: np.ndarray, sr: int) -> np.ndarray:
    mono = x.mean(axis=1) if x.ndim == 2 else x
    if sr != TARGET_SR:
        g = int(np.gcd(sr, TARGET_SR))
        mono = resample_poly(mono, TARGET_SR // g, sr // g)
    return np.asarray(mono, dtype="float32")


def audio_features(mono16k: np.ndarray, sr_in: int) -> dict:
    """Label-free signal features. They describe the channel, not the speech content."""
    n = len(mono16k)
    rms = float(np.sqrt(np.mean(mono16k**2))) if n else 0.0
    clip = float(np.mean(np.abs(mono16k) > 0.99)) if n else 0.0
    hf = 0.0
    if n >= 512:  # share of energy above 4 kHz: ~0 for telephone-band microphones
        spec = np.abs(np.fft.rfft(mono16k)) ** 2
        freqs = np.fft.rfftfreq(n, 1 / TARGET_SR)
        hf = float(spec[freqs > 4000].sum() / (spec.sum() + 1e-12))
    # Level of the loudest 5% of 25 ms frames: speech has loud frames even when quiet overall.
    fl = 400
    if n >= fl:
        frames = mono16k[: n - n % fl].reshape(-1, fl)
        frame_rms = np.sqrt(np.mean(frames**2, axis=1))
        active = float(np.percentile(frame_rms, 95))
        floor = float(np.percentile(frame_rms, 10))  # background noise level between words
    else:
        active = floor = rms
    return {"duration_s": round(n / TARGET_SR, 3), "rms_db": round(20 * np.log10(rms + 1e-9), 1),
            "active_db": round(20 * np.log10(active + 1e-9), 1),
            "snr_db": round(20 * np.log10((active + 1e-9) / (floor + 1e-9)), 1),
            "clip_frac": round(clip, 4), "hf_ratio": round(hf, 4), "sr_in": int(sr_in)}


def gate(data, cfg: dict | None = None):
    """Check one recording. Returns (GateResult, 16 kHz mono samples or None)."""
    cfg = {**DEFAULT_GATE, **(cfg or {})}
    try:
        x, sr, ch = read_audio(data)
    except Exception as e:  # corrupt file, wrong format, empty upload
        return GateResult(False, [f"unreadable audio ({type(e).__name__})"]), None
    if len(x) == 0:
        return GateResult(False, ["empty audio"]), None
    mono = to_model_input(x, sr)
    f = audio_features(mono, sr)
    f["channels"] = int(ch)
    res = GateResult(True, features=f)
    if f["duration_s"] < cfg["min_s"]:
        res.issues.append(f"too short ({f['duration_s']} s)")
    if f["duration_s"] > cfg["max_s"]:
        res.issues.append(f"too long ({f['duration_s']} s > {cfg['max_s']} s)")
    if f["active_db"] < cfg["silence_db"]:
        res.issues.append(f"silent or near-silent (loudest frames {f['active_db']} dBFS)")
    if f["clip_frac"] > cfg["max_clip"]:
        res.warnings.append(f"clipping ({f['clip_frac']:.1%} of samples)")
    if sr < cfg["min_sr"]:
        res.warnings.append(f"low sample rate ({sr} Hz): telephone-band audio")
    elif f["hf_ratio"] < cfg["min_hf_ratio"]:
        res.warnings.append("no energy above 4 kHz: narrow-band microphone or upsampled audio")
    if ch > 1:
        res.warnings.append(f"{ch} channels, downmixed to mono")
    res.ok = not res.issues
    return res, mono
