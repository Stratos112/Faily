"""How a voice sounds, in numbers — so a Piper training run can be graphed by more than its
loss. Model-free signal measures (no Whisper, no GPU), taken on:
  * a trained model: the pronunciation test's 16 sentences (pronunciation.SENTENCES), spoken
    by the piper command-line synthesizer on the CPU — cheap enough to run on every
    checkpoint while training is still going;
  * the clips a run trains on: the "target" each measure should move toward;
  * the base voice: where the run starts.
Pronunciation (Whisper) stays in modules/pronunciation.py.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np

SR = 22050                      # everything is measured at Piper's rate so the spectra line up
_LTAS_BINS = 64
TARGET_CLIPS = 40               # training clips sampled for the target values

# key -> (label, unit, what it is). The order is the order of the graphs.
METRICS: dict[str, tuple[str, str, str]] = {
    "match": ("VOICE MATCH", "%", "How closely the model's overall spectrum (its long-term average, roughly "
              "the timbre) matches your training clips. 100% would be identical; the target line is how much "
              "two halves of your own clips agree with each other."),
    "wps": ("SPEAKING RATE", "words/s", "Words per second of speech, edge silence excluded."),
    "pitch": ("PITCH", "Hz", "Median pitch of the voiced sounds."),
    "pitch_range": ("PITCH RANGE", "semitones", "How far the pitch moves (10th to 90th percentile) — "
                    "flat, monotone speech is low, lively speech high."),
    "centroid": ("BRIGHTNESS", "Hz", "Spectral centroid of the speech: higher is brighter / thinner, "
                 "lower darker / fuller."),
    "loudness": ("LOUDNESS", "dBFS", "Average level of the speech."),
    "noise": ("BACKGROUND", "dB", "Level of the quietest moments relative to the speech: room tone, hiss, "
              "hum. Further below 0 is quieter."),
}


def _sentences():
    from faily.modules.pronunciation import SENTENCES
    return SENTENCES


def _sentences_hash() -> str:
    return hashlib.sha1("\n".join(_sentences()).encode()).hexdigest()[:10]


# ── one clip ─────────────────────────────────────────────────────────────────

def measure_audio(x: np.ndarray, sr: int, text: str | None = None) -> dict | None:
    """The measures for one clip (mono or not, any rate). wps only when `text` is given.
    None for a clip too short or quiet to measure."""
    import librosa

    x = np.asarray(x, dtype=np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != SR:
        x = librosa.resample(x, orig_sr=sr, target_sr=SR)
    if len(x) < SR // 4:
        return None
    n_fft, hop = 1024, 256
    S = np.abs(librosa.stft(x, n_fft=n_fft, hop_length=hop)) ** 2
    energy = S.sum(axis=0)
    db = 10 * np.log10(energy + 1e-12)
    top = float(np.percentile(db, 95))
    active = db > top - 40.0
    if active.sum() < 5:
        return None
    idx = np.flatnonzero(active)
    span = (idx[-1] - idx[0] + 1) * hop / SR                  # speech, edge silence excluded
    freqs = librosa.fft_frequencies(sr=SR, n_fft=n_fft)
    loud = db > np.percentile(db[active], 30)
    M = np.sqrt(S)                                            # magnitude-weighted, as librosa's spectral_centroid
    frame_centroid = (M * freqs[:, None]).sum(axis=0) / (M.sum(axis=0) + 1e-12)
    centroid = float(frame_centroid[loud].mean())
    nonzero = energy[energy > 0]
    noise = 10 * np.log10(np.percentile(nonzero, 5) / (np.percentile(nonzero, 90) + 1e-12) + 1e-12)
    rms = librosa.feature.rms(y=x, frame_length=n_fft, hop_length=hop)[0][: len(active)]   # same frames as S
    loudness = float(20 * np.log10(np.sqrt((rms[active[: len(rms)]] ** 2).mean()) + 1e-12))   # dBFS of the speech
    ltas = np.log(S[:, active].mean(axis=1) + 1e-12)[: n_fft // 2]
    ltas = ltas[: len(ltas) // _LTAS_BINS * _LTAS_BINS].reshape(_LTAS_BINS, -1).mean(axis=1)
    x16 = librosa.resample(x, orig_sr=SR, target_sr=16000)
    f0, voiced, _ = librosa.pyin(x16, fmin=65, fmax=500, sr=16000, frame_length=1024, hop_length=256)
    f0 = f0[voiced & np.isfinite(f0)] if f0 is not None else np.array([])
    out = {
        "dur": round(span, 3), "centroid": round(centroid, 1), "noise": round(float(noise), 1),
        "loudness": round(loudness, 1), "ltas": (ltas - ltas.mean()).round(3).tolist(),
        "pitch": round(float(np.median(f0)), 1) if len(f0) >= 5 else None,
        "pitch_range": round(float(12 * np.log2(np.percentile(f0, 90) / np.percentile(f0, 10))), 2)
        if len(f0) >= 5 else None,
    }
    if text:
        out["words"] = len(text.split())
    return out


def summarize(clips: list[dict]) -> dict:
    """One value per measure over many clips (medians; wps = all words / all speech time;
    ltas = the average spectrum)."""
    clips = [c for c in clips if c]
    if not clips:
        return {}

    def med(k):
        vals = [c[k] for c in clips if c.get(k) is not None]
        return round(float(np.median(vals)), 2) if vals else None

    worded = [c for c in clips if c.get("words")]
    out = {k: med(k) for k in ("pitch", "pitch_range", "centroid", "loudness", "noise")}
    out["wps"] = round(float(sum(c["words"] for c in worded) / max(sum(c["dur"] for c in worded), 1e-6)), 2) if worded else None
    ltas = np.mean([c["ltas"] for c in clips], axis=0)
    out["ltas"] = (ltas - ltas.mean()).round(3).tolist()
    out["n"] = len(clips)
    return out


def voice_match(a: list[float] | None, b: list[float] | None) -> float | None:
    """0-100: cosine similarity of two (mean-removed) long-term spectra."""
    if not a or not b:
        return None
    va, vb = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    cos = float(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-12))
    return round(max(0.0, cos) * 100, 2)


# ── a model ──────────────────────────────────────────────────────────────────

def _cache_path(onnx: Path) -> Path:
    return onnx.with_suffix(".metrics.json")


def cached(onnx: Path) -> dict | None:
    try:
        d = json.loads(_cache_path(onnx).read_text())
        return d if d.get("sentences") == _sentences_hash() else None
    except (OSError, ValueError):
        return None


def measure_model(onnx: Path, use_cache: bool = True) -> dict:
    """summarize() of the model speaking the pronunciation test's sentences, cached beside the
    .onnx. Blocking: CPU synthesis (the piper binary) + signal analysis; call from a thread."""
    import soundfile as sf
    from faily.modules.piper import _piper_bin

    if use_cache and (hit := cached(onnx)):
        return hit
    sentences = _sentences()
    with tempfile.TemporaryDirectory() as tmp:
        r = subprocess.run(
            [str(_piper_bin()), "--model", str(onnx), "--config", str(onnx.with_suffix(".onnx.json")),
             "--output_dir", tmp],
            input=("\n".join(sentences) + "\n").encode(), capture_output=True,
        )
        wavs = sorted(Path(tmp).glob("*.wav"))
        if r.returncode != 0 or len(wavs) != len(sentences):
            raise RuntimeError(f"Piper produced {len(wavs)} of {len(sentences)} clips: {r.stderr.decode()[-300:]}")
        clips = []
        for text, wav in zip(sentences, wavs):
            x, sr = sf.read(str(wav), dtype="float32")
            clips.append(measure_audio(x, sr, text))
    out = {**summarize(clips), "sentences": _sentences_hash()}
    try:
        _cache_path(onnx).write_text(json.dumps(out))
    except OSError:
        pass
    return out


def measure_clips(clips: list[tuple[Path, str]], n: int = TARGET_CLIPS) -> dict:
    """summarize() of an even spread of up to `n` (audio, transcript) clips, plus
    "match_ceiling": how well two halves of them match each other (the best a model trained
    on them could be expected to reach on voice match)."""
    import soundfile as sf

    if not clips:
        return {}
    pick = [clips[int(i)] for i in np.unique(np.linspace(0, len(clips) - 1, min(n, len(clips))).astype(int))]
    measured = []
    for path, text in pick:
        try:
            x, sr = sf.read(str(path), dtype="float32")
        except Exception:
            continue
        measured.append(measure_audio(x, sr, text or None))
    measured = [m for m in measured if m]
    out = summarize(measured)
    if len(measured) >= 6:
        out["match_ceiling"] = voice_match(summarize(measured[0::2])["ltas"], summarize(measured[1::2])["ltas"])
    return out
