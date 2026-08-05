"""Voice style model — spectral + pitch transfer from reference clips."""
from pathlib import Path
import numpy as np
import soundfile as sf
import scipy.signal as sps
import scipy.ndimage

_SR = 22050
_N_FFT = 2048
_HOP = 512


# ── audio helpers ─────────────────────────────────────────────────────────────

def _load_mono(path: Path) -> np.ndarray:
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != _SR:
        audio = sps.resample_poly(audio, _SR, sr).astype(np.float32)
    return audio


def _mean_log_envelope(audio: np.ndarray) -> np.ndarray:
    """Time-averaged log-magnitude STFT envelope."""
    _, _, Zxx = sps.stft(audio, _SR, nperseg=_N_FFT, noverlap=_N_FFT - _HOP)
    return np.mean(np.log(np.abs(Zxx) + 1e-10), axis=1)


def _voiced_f0(audio: np.ndarray) -> np.ndarray:
    """Autocorrelation-based F0 detection; returns Hz of voiced frames."""
    frame = 512
    hop = 256
    lo, hi = int(_SR / 500), int(_SR / 60)
    out = []
    for i in range(0, len(audio) - frame, hop):
        w = audio[i:i + frame] * np.hanning(frame)
        if np.sqrt(np.mean(w ** 2)) < 0.005:
            continue
        c = np.correlate(w, w, mode="full")[frame - 1:]
        c /= c[0] + 1e-10
        if hi < len(c):
            idx = np.argmax(c[lo:hi]) + lo
            if c[idx] > 0.25:
                out.append(_SR / idx)
    return np.array(out, dtype=np.float32)


def _shift_pitch(audio: np.ndarray, semitones: float) -> np.ndarray:
    if semitones == 0:
        return audio
    factor = 2 ** (semitones / 12)
    shifted = sps.resample(audio, max(1, int(len(audio) / factor)))
    return sps.resample(shifted, len(audio)).astype(np.float32)


def _prog(ref, val):
    if ref is not None:
        ref[0] = float(val)


# ── training ──────────────────────────────────────────────────────────────────

def train_rvc(
    char_name: str,
    clips: list[Path],
    out_dir: Path,
    progress_ref: list[float] | None = None,
) -> Path:
    """Build a compact voice profile (.npz) from reference clips."""
    if not clips:
        raise ValueError("No reference clips to train from")

    _prog(progress_ref, 0.05)
    envs, f0s = [], []

    for i, clip in enumerate(clips):
        audio = _load_mono(clip)
        envs.append(_mean_log_envelope(audio))
        f0 = _voiced_f0(audio)
        if len(f0):
            f0s.extend(f0.tolist())
        _prog(progress_ref, 0.1 + 0.78 * (i + 1) / len(clips))

    mean_env = np.mean(envs, axis=0)
    f0_arr = np.array(f0s) if f0s else np.array([150.0])
    log_f0 = np.log(f0_arr)

    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "rvc_model.npz"
    np.savez(
        str(model_path),
        mean_env=mean_env,
        f0_mean=np.array([log_f0.mean()]),
        f0_std=np.array([log_f0.std()]),
        n_clips=np.array([len(clips)]),
    )
    _prog(progress_ref, 1.0)
    return model_path


# ── inference ─────────────────────────────────────────────────────────────────

def speak_generate(
    text: str,
    model_path: str,
    output_dir: Path,
    pitch_shift: int = 0,
    index_rate: float = 0.75,
    filter_radius: int = 3,
    char_name: str | None = None,
) -> Path:
    """TTS → voice-style transfer → output wav."""
    audio = _tts(text)

    model = np.load(str(model_path))
    target_env = model["mean_env"]

    # Auto pitch-match to the character's trained F0 when user hasn't set a manual shift
    if pitch_shift == 0 and "f0_mean" in model:
        tts_f0 = _voiced_f0(audio)
        if len(tts_f0):
            tts_log_f0 = np.log(tts_f0).mean()
            target_log_f0 = float(model["f0_mean"][0])
            auto_semitones = (target_log_f0 - tts_log_f0) * 12.0 / np.log(2)
            audio = _shift_pitch(audio, auto_semitones)

    if index_rate > 0:
        audio = _envelope_transfer(audio, target_env, index_rate, filter_radius)

    if pitch_shift != 0:
        audio = _shift_pitch(audio, pitch_shift)

    output_dir.mkdir(parents=True, exist_ok=True)
    slug = text[:28].strip().replace(" ", "_").replace("/", "-")
    out_path = output_dir / f"speak_{char_name or 'char'}_{slug}.wav"
    sf.write(str(out_path), audio, _SR)
    return out_path


def _envelope_transfer(
    audio: np.ndarray,
    target_env: np.ndarray,
    rate: float,
    filter_radius: int,
) -> np.ndarray:
    """Shift the spectral envelope of audio toward target_env."""
    _, _, Zxx = sps.stft(audio, _SR, nperseg=_N_FFT, noverlap=_N_FFT - _HOP)
    src_env = np.mean(np.log(np.abs(Zxx) + 1e-10), axis=1)

    n = min(len(src_env), len(target_env))
    delta = (target_env[:n] - src_env[:n]) * rate

    win = filter_radius * 4 + 1
    if filter_radius > 0 and win >= 3 and win < n:
        delta = sps.savgol_filter(delta, win, 2)

    Zxx[:n] *= np.exp(delta)[:, np.newaxis]

    _, out = sps.istft(Zxx, _SR, nperseg=_N_FFT, noverlap=_N_FFT - _HOP)
    out = out.astype(np.float32)

    rms_in = np.sqrt(np.mean(audio ** 2)) + 1e-10
    rms_out = np.sqrt(np.mean(out ** 2)) + 1e-10
    return (out * rms_in / rms_out).astype(np.float32)


def _tts(text: str) -> np.ndarray:
    """Generate neutral speech via MMS-TTS."""
    from transformers import pipeline as hf_pipeline
    pipe = hf_pipeline("text-to-speech", model="facebook/mms-tts-eng")
    result = pipe(text)
    audio = np.array(result["audio"]).squeeze().astype(np.float32)
    sr = result["sampling_rate"]
    if sr != _SR:
        audio = sps.resample_poly(audio, _SR, sr).astype(np.float32)
    return audio
