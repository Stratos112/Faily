import soundfile as sf
import numpy as np
from pathlib import Path


def audio_info(path: Path) -> dict:
    info = sf.info(str(path))
    return {"duration": info.duration, "sample_rate": info.samplerate, "channels": info.channels}


def apply_edits(
    src: Path,
    out: Path,
    volume_db: float = 0.0,
    speed: float = 1.0,
    pitch_semitones: int = 0,
    trim_start: float = 0.0,
    trim_end: float = 0.0,
    trim_silence: bool = False,
    stereo: bool = False,
) -> Path:
    data, sr = sf.read(str(src), dtype="float32", always_2d=False)
    n = data.shape[0]

    # trim edges
    s = min(int(trim_start * sr), n)
    e = max(n - int(trim_end * sr), s + 1)
    data = data[s:e]

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
        import torch
        import torchaudio.functional as F
        mono = data.ndim == 1
        t = torch.from_numpy(data[np.newaxis] if mono else data.T.copy())
        t = F.pitch_shift(t, sr, n_steps=float(pitch_semitones))
        data = t.squeeze(0).numpy() if mono else t.T.numpy()

    # stereo — copy mono to both channels
    if stereo and data.ndim == 1:
        data = np.stack([data, data], axis=1)

    if len(data) == 0:
        data = np.zeros(sr, dtype=np.float32)

    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out), data, sr)
    return out
