"""Whisper-family ASR engine on CPU (Hugging Face transformers)."""
from __future__ import annotations

import io
import logging
import os
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import yaml
from transformers import WhisperForConditionalGeneration, WhisperProcessor

log = logging.getLogger("asr.engine")
ROOT = Path(__file__).resolve().parents[2]
DECODE = yaml.safe_load((ROOT / "configs" / "params.yaml").read_text(encoding="utf-8"))["decode"]


def default_threads() -> int:
    return int(os.environ.get("ASR_THREADS", max(1, (os.cpu_count() or 2) // 2)))


class WhisperEngine:
    def __init__(self, model_id: str, quantize: bool = False, threads: int | None = None):
        torch.set_num_threads(threads or default_threads())
        t0 = time.time()
        self.model_id = model_id
        self.processor = WhisperProcessor.from_pretrained(model_id)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_id).eval()
        self.quantized = quantize
        self.last_cut: list[bool] = []
        if quantize:  # int8 weights for Linear layers: smaller and usually faster on CPU
            with warnings.catch_warnings():  # torch announces a future API change; the lab pins torch
                warnings.simplefilter("ignore", UserWarning)
                self.model = torch.ao.quantization.quantize_dynamic(
                    self.model, {torch.nn.Linear}, dtype=torch.qint8)
        self.load_s = round(time.time() - t0, 1)
        log.info("model loaded %s quantized=%s in %.1fs", model_id, quantize, self.load_s)

    def size_mb(self) -> float:
        buf = io.BytesIO()
        torch.save(self.model.state_dict(), buf)
        return round(buf.tell() / 1e6, 1)

    @torch.inference_mode()
    def transcribe(self, batch: list[np.ndarray], max_new_tokens: int | None = None) -> list[str]:
        limit = max_new_tokens or DECODE["max_new_tokens"]
        feats = self.processor(batch, sampling_rate=16000, return_tensors="pt", return_attention_mask=True)
        ids = self.model.generate(feats.input_features, attention_mask=feats.attention_mask,
                                  language="am", task="transcribe",
                                  max_new_tokens=limit, num_beams=DECODE["num_beams"])
        # An output that used all `limit` tokens without an end-of-text token was cut off: the limit
        # is too low, or the model is looping (repeating itself). Either way the text is incomplete.
        eos = self.model.generation_config.eos_token_id
        self.last_cut = [ids.shape[1] >= limit and bool((row != eos).all()) for row in ids]
        return [t.strip() for t in self.processor.batch_decode(ids, skip_special_tokens=True)]
