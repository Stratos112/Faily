import soundfile as sf
import numpy as np
from pathlib import Path


def audio_info(path: Path) -> dict:
    info = sf.info(str(path))
    return {"duration": info.duration, "sample_rate": info.samplerate, "channels": info.channels}


def ensure_stereo(path: Path):
    """Duplicate a mono WAV to stereo (L=R) in-place. No-op if already stereo."""
    data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if data.ndim == 1:
        sf.write(str(path), np.stack([data, data], axis=1), sr)


def _highpass(data: np.ndarray, sr: int, cutoff_hz: float) -> np.ndarray:
    """4th-order Butterworth high-pass — cuts rumble/hum below cutoff_hz.
    Pure filtering, no noise estimation, so it's safe/predictable on any input."""
    import scipy.signal as sps
    sos = sps.butter(4, cutoff_hz, btype="highpass", fs=sr, output="sos")
    if data.ndim == 1:
        return sps.sosfiltfilt(sos, data).astype(np.float32)
    return np.stack(
        [sps.sosfiltfilt(sos, data[:, c]) for c in range(data.shape[1])], axis=1
    ).astype(np.float32)


def _denoise_channel(x: np.ndarray, sr: int, strength: float) -> np.ndarray:
    """Spectral-gating noise reduction. Blindly estimating a noise profile from
    the whole clip performs poorly (verified: it attenuates speech and noise by
    roughly the same amount, barely improving effective SNR). Finding the
    quietest short window in the clip and using *that* as the noise reference
    instead gives noise-floor reduction several times larger than the speech
    attenuation — that's the difference this function exists to apply."""
    import noisereduce as nr

    win = max(int(0.2 * sr), 1)
    noise_clip = None
    if len(x) > win * 3:
        hop = max(win // 2, 1)
        n_wins = max((len(x) - win) // hop, 1)
        energies = [
            float(np.sqrt(np.mean(x[i * hop: i * hop + win] ** 2)))
            for i in range(n_wins)
        ]
        q = int(np.argmin(energies))
        noise_clip = x[q * hop: q * hop + win]
    return nr.reduce_noise(
        y=x, sr=sr, y_noise=noise_clip, stationary=True, prop_decrease=strength,
    ).astype(np.float32)


def apply_edits(
    src: Path,
    out: Path,
    volume_db: float = 0.0,
    speed: float = 1.0,
    pitch_semitones: float = 0.0,
    trim_start: float = 0.0,
    trim_end: float = 0.0,
    trim_silence: bool = False,
    stereo: bool = False,
    mono: bool = False,
    highpass: bool = False,
    highpass_hz: float = 80.0,
    denoise: bool = False,
    denoise_strength: float = 0.8,
) -> Path:
    data, sr = sf.read(str(src), dtype="float32", always_2d=False)
    n = data.shape[0]

    # trim edges
    s = min(int(trim_start * sr), n)
    e = max(n - int(trim_end * sr), s + 1)
    data = data[s:e]

    # high-pass + denoise run early, before amplitude-based silence trimming,
    # so a hummy/noisy noise floor doesn't throw off that threshold.
    if highpass and len(data) > 0:
        data = _highpass(data, sr, highpass_hz)

    if denoise and len(data) > 0:
        if data.ndim == 1:
            data = _denoise_channel(data, sr, denoise_strength)
        else:
            data = np.stack(
                [_denoise_channel(data[:, c], sr, denoise_strength) for c in range(data.shape[1])],
                axis=1,
            )

    # trim silence via amplitude threshold
    if trim_silence and len(data) > 0:
        mono = data if data.ndim == 1 else data.mean(axis=1)
        mask = np.abs(mono) > 0.02
        if mask.any():
            first = int(np.argmax(mask))
            last = int(len(mask) - np.argmax(mask[::-1]))
            data = data[first:last]

    # volume
    if abs(volume_db) > 0.01:
        data = np.clip(data * (10.0 ** (volume_db / 20.0)), -1.0, 1.0)

    # speed — tape-speed effect (changes pitch proportionally)
    if abs(speed - 1.0) > 0.001 and len(data) > 0:
        import torch
        import torchaudio.functional as F
        mono = data.ndim == 1
        t = torch.from_numpy(data[np.newaxis] if mono else data.T.copy())
        t = F.resample(t, int(sr * speed), sr)
        data = t.squeeze(0).numpy() if mono else t.T.numpy()

    # pitch shift — independent of speed, preserves duration
    if pitch_semitones != 0 and len(data) > 0:
        import librosa
        if data.ndim == 1:
            data = librosa.effects.pitch_shift(data, sr=sr, n_steps=float(pitch_semitones)).astype(np.float32)
        else:
            channels = [
                librosa.effects.pitch_shift(data[:, c], sr=sr, n_steps=float(pitch_semitones)).astype(np.float32)
                for c in range(data.shape[1])
            ]
            data = np.stack(channels, axis=1)

    # stereo — copy mono to both channels
    if stereo and data.ndim == 1:
        data = np.stack([data, data], axis=1)

    # mono — average stereo channels down
    if mono and data.ndim > 1:
        data = data.mean(axis=1)

    if len(data) == 0:
        data = np.zeros(sr, dtype=np.float32)

    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out), data, sr)
    return out


def mix_tracks(tracks: list[dict], out: Path) -> Path:
    """tracks: [{"path": Path|None, "vol": float, "muted": bool, "offset": float}]
    offset is in seconds; clips are mixed starting at that position."""
    segments: list[tuple[int, np.ndarray]] = []  # (offset_samples, data)
    target_sr: int | None = None

    for t in tracks:
        if t.get("muted") or not t.get("path"):
            continue
        data, sr = sf.read(str(t["path"]), dtype="float32", always_2d=True)
        if target_sr is None:
            target_sr = sr
        elif sr != target_sr:
            import torch
            import torchaudio.functional as F
            wav = torch.from_numpy(data.T.copy())
            data = F.resample(wav, sr, target_sr).T.numpy()
        data = data * float(t.get("vol", 1.0))
        offset_samples = int(t.get("offset", 0.0) * target_sr)
        segments.append((offset_samples, data))

    if not segments or target_sr is None:
        raise ValueError("No unmuted tracks to mix")

    max_end = max(off + a.shape[0] for off, a in segments)
    max_ch  = max(a.shape[1] for _, a in segments)
    mixed = np.zeros((max_end, max_ch), dtype=np.float32)

    for off, a in segments:
        n, c = a.shape
        if c < max_ch:
            a = np.tile(a, (1, max_ch // c))
        mixed[off:off + n] += a

    peak = np.abs(mixed).max()
    if peak > 1.0:
        mixed /= peak

    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out), mixed if max_ch > 1 else mixed[:, 0], target_sr)
    return out
