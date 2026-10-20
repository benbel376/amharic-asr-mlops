"""What does the model say when nobody speaks? (Whisper models can "hallucinate" text.)

    python scripts/silence_probe.py

Feeds silence, low noise and a clinic-like hum straight to the model (no audio gate).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from asr.audio import gate  # noqa: E402
from asr.engine import WhisperEngine  # noqa: E402

P = yaml.safe_load((ROOT / "configs" / "params.yaml").read_text(encoding="utf-8"))


def main() -> None:
    sr, rng = 16000, np.random.default_rng(0)
    t = np.arange(5 * sr) / sr
    probes = {
        "digital silence (5 s)": np.zeros(5 * sr, "float32"),
        "room noise (5 s)": rng.normal(0, 0.003, 5 * sr).astype("float32"),
        "fan hum 100 Hz (5 s)": (0.02 * np.sin(2 * np.pi * 100 * t)).astype("float32"),
    }
    eng = WhisperEngine(str(ROOT / P["models"]["champion"]["path"]))
    for name, x in probes.items():
        import io

        import soundfile as sf
        buf = io.BytesIO()
        sf.write(buf, x, sr, format="WAV")
        g, _ = gate(buf.getvalue())
        text = eng.transcribe([x])[0]
        print(f"{name:24s} model says: {text[:70]!r:74s} gate: {'PASS' if g.ok else 'REJECT ' + '; '.join(g.issues)}")


if __name__ == "__main__":
    main()
