"""Pronunciation score for a trained Piper model: synthesize a fixed set of
sentences the model never trained on, transcribe them with Whisper, and count
how many words come back right. Fine-tuning on a small dataset can scramble
pronunciation long before the voice sounds wrong, and this catches it
objectively (an untouched base voice scores ~100%).
"""
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

SENTENCES = [
    "The weather this morning was cold, but the afternoon turned out warm and bright.",
    "She picked up the phone, looked at the screen, and decided not to answer it.",
    "Please remember to bring your passport, your tickets, and a warm jacket.",
    "I honestly cannot believe how quickly the whole summer has gone by.",
    "Could you tell me where the nearest pharmacy is, and whether it's still open?",
    "We walked along the river until the sun went down behind the hills.",
    "He said the meeting would start at nine, although nobody believed him.",
    "There are three reasons for this: cost, time, and a lack of interest.",
    "The train to the coast leaves at half past seven every evening.",
    "My brother lent me his car, but he wants it back before Friday.",
    "After dinner, they sat on the porch and talked until it got dark.",
    "It took longer than expected, mostly because the roads were closed.",
    "Do you remember the little cafe where we first met last spring?",
    "The package arrived yesterday, though the box was slightly damaged.",
    "Nobody told me the instructions had changed, so I did it the old way.",
    "A gentle breeze moved through the tall grass beside the quiet lake.",
]


def _sentences_hash() -> str:
    return hashlib.sha1("\n".join(SENTENCES).encode()).hexdigest()[:10]


def _score_path(onnx: Path) -> Path:
    return onnx.with_suffix(".score.json")


def cached_score(onnx: Path) -> dict | None:
    try:
        d = json.loads(_score_path(onnx).read_text())
        return d if d.get("sentences") == _sentences_hash() else None
    except (OSError, ValueError):
        return None


def score_model(onnx: Path, use_cache: bool = True) -> dict:
    """{"accuracy": 0-1, "words_per_sec": float, "worst": [{"said","heard"}...]}.
    Blocking (synthesis + Whisper); call from a worker thread."""
    from faily.modules import cleanup as K
    from faily.modules.asr import transcribe
    from faily.modules.piper import _piper_bin

    if use_cache and (hit := cached_score(onnx)):
        return hit
    cfg = onnx.with_suffix(".onnx.json")
    with tempfile.TemporaryDirectory() as tmp:
        r = subprocess.run(
            [str(_piper_bin()), "--model", str(onnx), "--config", str(cfg), "--output_dir", tmp,
             "--sentence-silence", "0.3"],
            input=("\n".join(SENTENCES) + "\n").encode(), capture_output=True,
        )
        wavs = sorted(Path(tmp).glob("*.wav"))
        if r.returncode != 0 or len(wavs) != len(SENTENCES):
            raise RuntimeError(f"Piper produced {len(wavs)} of {len(SENTENCES)} test clips: {r.stderr.decode()[-300:]}")
        errors, words, secs, worst = [], 0, 0.0, []
        for text, wav in zip(SENTENCES, wavs):
            x, sr = sf.read(str(wav), dtype="float32")
            heard = transcribe(x.mean(axis=1) if x.ndim > 1 else x, sr)
            ref_w, hyp_w = K.normalize_words(text), K.normalize_words(heard)
            wer = K.word_error_rate(ref_w, hyp_w)
            errors.append(wer * len(ref_w))
            words += len(ref_w)
            secs += len(x) / sr
            if wer > 0.15:
                worst.append({"said": text, "heard": heard, "wer": round(wer, 2)})
    out = {
        "accuracy": round(max(0.0, 1 - sum(errors) / max(words, 1)), 3),
        "words_per_sec": round(words / max(secs, 1e-6), 2),
        "worst": sorted(worst, key=lambda w: -w["wer"])[:4],
        "sentences": _sentences_hash(),
    }
    try:
        _score_path(onnx).write_text(json.dumps(out, indent=1))
    except OSError:
        pass
    return out
