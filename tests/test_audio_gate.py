import io

import numpy as np
import soundfile as sf

from asr.audio import gate


def wav(x, sr=16000):
    buf = io.BytesIO()
    sf.write(buf, x, sr, format="WAV")
    return buf.getvalue()


def speechlike(seconds=2.0, sr=16000):
    rng = np.random.default_rng(0)
    return (0.1 * rng.standard_normal(int(seconds * sr))).astype("float32")  # broadband


def test_good_audio_passes():
    g, mono = gate(wav(speechlike()))
    assert g.ok and not g.issues and mono is not None and len(mono) == 32000


def test_silence_rejected():
    g, _ = gate(wav(np.zeros(32000, dtype="float32")))
    assert not g.ok and any("silent" in i for i in g.issues)


def test_too_short_rejected():
    g, _ = gate(wav(speechlike(0.2)))
    assert not g.ok and any("short" in i for i in g.issues)


def test_corrupt_and_not_audio_rejected():
    assert not gate(b"RIFF\x00\x00\x00\x00WAVEfmt " + bytes(20))[0].ok
    assert not gate(b"hello, this is text")[0].ok
    assert not gate(b"")[0].ok


def test_low_sample_rate_warns_and_resamples():
    g, mono = gate(wav(speechlike(sr=8000), sr=8000))
    assert g.ok and any("sample rate" in w for w in g.warnings)
    assert len(mono) == 32000  # resampled to 16 kHz


def test_stereo_downmixed_and_clipping_flagged():
    x = np.clip(speechlike() * 20, -1, 1)
    g, mono = gate(wav(np.stack([x, x], axis=1)))
    assert g.ok and mono.ndim == 1
    assert any("channels" in w for w in g.warnings) and any("clipping" in w for w in g.warnings)


def test_features_contain_no_samples():
    g, _ = gate(wav(speechlike()))
    assert set(g.features) == {"duration_s", "rms_db", "active_db", "snr_db", "clip_frac", "hf_ratio", "sr_in", "channels"}
