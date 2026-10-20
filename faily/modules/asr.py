"""Whisper transcription for dataset cleanup and the pronunciation score.

Calls the Whisper model directly (processor + generate) instead of going through
transformers' ASR pipeline: transformers 5's pipeline decodes through
torchcodec, whose native libraries are the fragile part of this stack, and we
always already have decoded samples anyway.
"""
import numpy as np

from faily.core.model_manager import manager

MODEL_ID = "openai/whisper-large-v3-turbo"
_CHUNK_SECS = 28.0       # Whisper's window is 30 s


def _load():
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    dtype = torch.float16 if manager.device == "cuda" else torch.float32

    def load(local_only: bool):
        proc = WhisperProcessor.from_pretrained(MODEL_ID, local_files_only=local_only)
        model = WhisperForConditionalGeneration.from_pretrained(
            MODEL_ID, torch_dtype=dtype, local_files_only=local_only)
        return proc, model

    try:
        proc, model = load(True)          # cached: no network at all (and no Hub warning)
    except OSError:
        proc, model = load(False)         # first run: download it
    return proc, model.to(manager.device).eval()


def _resample_16k(audio: np.ndarray, sr: int) -> np.ndarray:
    import scipy.signal as sps

    if sr == 16000:
        return audio.astype("float32")
    g = np.gcd(sr, 16000)
    return sps.resample_poly(audio, 16000 // g, sr // g).astype("float32")


def transcribe(audio: np.ndarray, sr: int) -> str:
    """Transcribe mono float samples. Blocking; call from a worker thread."""
    import torch

    proc, model = manager.load("whisper_asr", _load)
    audio = _resample_16k(audio.mean(axis=1) if audio.ndim > 1 else audio, sr)
    step = int(_CHUNK_SECS * 16000)
    texts = []
    for i in range(0, max(len(audio), 1), step):
        chunk = audio[i:i + step]
        if len(chunk) < 1600:      # < 0.1 s tail: nothing to transcribe
            continue
        feats = proc(chunk, sampling_rate=16000, return_tensors="pt").input_features
        feats = feats.to(manager.device, dtype=model.dtype)
        with torch.no_grad():
            try:
                ids = model.generate(feats, language="en", task="transcribe", max_new_tokens=220)
            except ValueError:       # outdated generation config: let it auto-detect
                ids = model.generate(feats, max_new_tokens=220)
        texts.append(proc.tokenizer.decode(ids[0].tolist(), skip_special_tokens=True).strip())
    return " ".join(t for t in texts if t)


def unload() -> None:
    manager.unload("whisper_asr")
