"""Split a long recording into clip-sized slices at natural pauses.

Used by the CLONE tab's "split & qualify" import: the slices become a character's
reference clips. Cuts go in the middle of pauses, so every slice keeps some
silence at both ends (the trim check later tightens it), and a cut never lands
inside a word unless a stretch has no pause at all for far longer than a slice
should be — then it falls on the quietest point available.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from faily.modules.qualify import _frame_db

HOP = 0.01                       # seconds per analysis frame (matches qualify._frame_db)

# slice length presets: (target, min, max) seconds
PRESETS = {
    "short":  (4.0, 1.5, 8.0),
    "medium": (7.0, 2.5, 12.0),
    "long":   (11.0, 4.0, 15.0),      # F5-TTS prefers references up to ~15 s
}


@dataclass
class Slice:
    start: float
    end: float
    speech: float                # seconds of actual speech inside

    @property
    def duration(self) -> float:
        return self.end - self.start


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    out, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            out.append((i, j + 1))
            i = j + 1
        else:
            i += 1
    return out


def find_slices(
    x: np.ndarray, sr: int, *, target: float = 7.0, min_len: float = 2.5, max_len: float = 12.0,
    min_pause: float = 0.25,
) -> list[Slice]:
    """Slice boundaries (in seconds) for mono audio `x`."""
    db = _frame_db(x, sr)
    n = len(db)
    if n < 10:
        return []
    floor, top = float(np.percentile(db, 10)), float(np.percentile(db, 95))
    active = db > max(floor + 10.0, top - 30.0)
    # a few frames of closing: stop-consonant gaps inside words aren't pauses
    for s, e in _runs(~active):
        if e - s < 6 and s > 0 and e < n:
            active[s:e] = True
    speech = _runs(active)
    if not speech:
        return []
    first, last = speech[0][0], speech[-1][1]

    # candidate cut points: middle of each pause between speech (pause length in seconds)
    pauses = []
    for (_, e0), (s1, _) in zip(speech, speech[1:]):
        length = (s1 - e0) * HOP
        if length >= 0.12:
            pauses.append(((e0 + s1) / 2 * HOP, length))
    energy = db                  # for forced cuts

    def speech_in(a: float, b: float) -> float:
        fa, fb = int(a / HOP), int(b / HOP)
        return float(active[fa:fb].sum() * HOP)

    cuts: list[float] = []
    s0 = max(first * HOP - 0.15, 0.0)
    end_t = min(last * HOP + 0.15, n * HOP)
    while end_t - s0 > max_len:
        window = [(c, ln) for c, ln in pauses if s0 + min_len <= c <= s0 + max_len]
        strong = [(c, ln) for c, ln in window if ln >= min_pause]
        pool = strong or window
        if pool:
            # prefer long pauses, then a slice length close to the target
            cut = max(pool, key=lambda p: p[1] * 2.0 - abs((p[0] - s0) - target) / target)[0]
        else:
            # no pause at all for max_len seconds: cut at the quietest frame in the late window
            lo, hi = int((s0 + max_len * 0.6) / HOP), int((s0 + max_len) / HOP)
            cut = (lo + int(np.argmin(energy[lo:hi]))) * HOP
        cuts.append(cut)
        s0 = cut
    bounds = [max(first * HOP - 0.15, 0.0)] + cuts + [end_t]

    # Pause-aligned cuts are only made once a slice would exceed max_len, so slices
    # run long. Rebalance: also cut early at a strong pause once past `target`.
    refined = [bounds[0]]
    for a, b in zip(bounds, bounds[1:]):
        start = a
        while b - start > target * 1.35:
            cand = [(c, ln) for c, ln in pauses if start + max(min_len, target * 0.6) <= c <= min(start + target * 1.35, b - min_len)
                    and ln >= min_pause]
            if not cand:
                break
            c = max(cand, key=lambda p: p[1] * 2.0 - abs((p[0] - start) - target) / target)[0]
            refined.append(c)
            start = c
        refined.append(b)
    bounds = refined

    slices = [Slice(a, b, speech_in(a, b)) for a, b in zip(bounds, bounds[1:])]
    # fold a too-short slice into a neighbour when the result still fits
    merged: list[Slice] = []
    for sl in slices:
        if merged and sl.duration < min_len and merged[-1].duration + sl.duration <= max_len + 2.0:
            p = merged.pop()
            merged.append(Slice(p.start, sl.end, p.speech + sl.speech))
        else:
            merged.append(sl)
    if len(merged) > 1 and merged[0].duration < min_len and merged[0].duration + merged[1].duration <= max_len + 2.0:
        a, b = merged[0], merged.pop(1)
        merged[0] = Slice(a.start, b.end, a.speech + b.speech)
    return [s for s in merged if s.speech >= 0.5]


def split_file(
    path: Path, out_dir: Path, preset: str = "medium", prefix: str = "slice",
) -> list[tuple[Path, Slice]]:
    """Slice `path` and write each slice as 16-bit mono WAV into out_dir.
    Returns [(wav_path, Slice)] in order."""
    target, min_len, max_len = PRESETS[preset]
    x, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    out_dir.mkdir(parents=True, exist_ok=True)
    fade = int(0.005 * sr)
    out = []
    for i, sl in enumerate(find_slices(x, sr, target=target, min_len=min_len, max_len=max_len), 1):
        a, b = int(sl.start * sr), min(int(sl.end * sr), len(x))
        piece = x[a:b].copy()
        if len(piece) > 2 * fade:                       # tiny fades: no clicks at forced cuts
            piece[:fade] *= np.linspace(0, 1, fade)
            piece[-fade:] *= np.linspace(1, 0, fade)
        wav = out_dir / f"{prefix}_{i:03d}.wav"
        sf.write(str(wav), piece, sr, subtype="PCM_16")
        out.append((wav, sl))
    return out
