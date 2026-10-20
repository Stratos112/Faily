"""Audit for generated ("padding") clips: find noises, artifacts and words that aren't
in the script the clip was generated from.

Padding clips are zero-shot TTS output of a known script line, so — unlike a
human recording — the transcript is the ground truth and anything else in the
audio is a defect: breaths and background noise, stray words ("Thank you."),
dropped words, a noisy floor.

Noise events are found from the signal, not from loudness: every real word contains
voiced (pitched) speech, so a stretch of audible sound with no pitch that stands
apart from the words is a breath, hiss or click. Whisper is used only to compare
words with the script and to verify a proposed trim by re-transcribing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf

HOP = 0.01
WIN = 0.032

# ── tunables ─────────────────────────────────────────────────────────────────
VOICED = 0.50                    # normalized autocorrelation peak that counts as pitched
MIN_EVENT = 0.10                 # seconds of unpitched sound before it is an event
MERGE_GAP = 0.06                 # bridge gaps this short (stop closures) between sound runs
EXCESS_DB = 6.0                  # mean dB a frame rises above the clip's steady background (per band)
NOISY_FLOOR_DB = -50.0           # a "silent" floor louder than this is background noise


@dataclass
class Event:
    start: float
    end: float
    kind: str                    # "breath" | "noise burst" | "hiss" | "click"
    level_db: float              # relative to the clip's voiced-speech level
    where: str                   # "start" | "end" | "inside"

    @property
    def duration(self) -> float:
        return self.end - self.start

    def label(self) -> str:
        return f"{self.kind} {self.start:.1f}–{self.end:.1f} s ({self.where})"


def _read(path: Path) -> tuple[np.ndarray, int]:
    x, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x, sr


def frame_features(x: np.ndarray, sr: int) -> dict[str, np.ndarray]:
    """Per 10 ms frame: level (dB), spectral flatness, pitch strength (0..1)."""
    w, h = int(WIN * sr), int(HOP * sr)
    if len(x) < w:
        x = np.pad(x, (0, w - len(x)))
    n = 1 + (len(x) - w) // h
    idx = np.arange(w)[None, :] + h * np.arange(n)[:, None]
    fr = x[idx]
    rms = 20 * np.log10(np.sqrt(np.mean(fr ** 2, axis=1)) + 1e-9)
    win = fr * np.hanning(w)
    P = np.abs(np.fft.rfft(win, axis=1)) ** 2 + 1e-12
    flat = np.exp(np.mean(np.log(P), axis=1)) / np.mean(P, axis=1)
    # pitch strength: peak of the normalized autocorrelation between 70 and 400 Hz
    size = 1 << int(np.ceil(np.log2(2 * w)))
    ac = np.fft.irfft(np.abs(np.fft.rfft(win, size, axis=1)) ** 2, axis=1)[:, :w]
    lo, hi = max(int(sr / 400), 2), min(int(sr / 70), w - 1)
    harm = ac[:, lo:hi].max(axis=1) / (ac[:, 0] + 1e-12)
    # How far each frame rises above the clip's steady background, per frequency band.
    # The background (hiss, hum, room tone) is taken from the quietest frames; judging
    # band by band finds a soft breath that overall loudness would lose in the hiss.
    spec = np.abs(np.fft.rfft(win, axis=1)) ** 2
    freqs = np.fft.rfftfreq(w, 1 / sr)
    edges = np.geomspace(100, min(10000, sr / 2 - 1), 8)
    bands = np.stack([spec[:, (freqs >= a) & (freqs < b)].mean(axis=1) + 1e-12
                      for a, b in zip(edges[:-1], edges[1:])], axis=1)
    bdb = 10 * np.log10(bands)
    quiet = rms <= np.percentile(rms, 15)
    profile = np.median(bdb[quiet], axis=0)
    excess = np.clip(bdb - profile, 0, None).mean(axis=1)
    return {"rms": rms, "flat": flat, "harm": np.clip(harm, 0, 1), "excess": excess}


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


def find_noise_events(x: np.ndarray, sr: int) -> tuple[list[Event], dict]:
    """Breaths, hiss, clicks and noise bursts in `x`, plus clip-level facts
    ({"floor_db", "speech_db", "snr_db", "voiced_start", "voiced_end"})."""
    f = frame_features(x, sr)
    rms, harm, flat = f["rms"], f["harm"], f["flat"]
    n = len(rms)
    voiced = (harm > VOICED) & (rms > np.percentile(rms, 30))
    if voiced.sum() < 5:
        return [], {"floor_db": float(np.percentile(rms, 5)), "speech_db": float(rms.max()),
                    "snr_db": 0.0, "voiced_start": 0.0, "voiced_end": n * HOP, "no_speech": True}
    speech_db = float(np.median(rms[voiced]))
    floor = float(np.percentile(rms, 5))
    audible = (f["excess"] > EXCESS_DB) & (rms > -80.0)
    # close short gaps so a stop closure inside a word doesn't split it
    for s, e in _runs(~audible):
        if e - s <= int(MERGE_GAP / HOP) and s > 0 and e < n:
            audible[s:e] = True
    vi = np.flatnonzero(voiced)
    v_start, v_end = vi[0], vi[-1] + 1
    events: list[Event] = []
    for s, e in _runs(audible):
        dur = (e - s) * HOP
        if dur < 0.02:
            continue
        pitched = float(voiced[s:e].mean())
        if pitched >= 0.08:                        # contains real voicing: it is (part of) a word
            continue
        level = float(rms[s:e].mean() - speech_db)
        if dur >= 0.8:
            kind = "hiss"
        elif dur < MIN_EVENT:
            # a click/pop: short, sharply above the background, not tonal
            if float(f["excess"][s:e].max()) < 15.0 or float(flat[s:e].mean()) < 0.2:
                continue
            kind = "click"
        else:
            kind = "noise burst" if level >= -12 else "breath"
        where = "start" if e <= v_start else ("end" if s >= v_end else "inside")
        events.append(Event(s * HOP, e * HOP, kind, round(level, 1), where))
    info = {
        "floor_db": floor, "speech_db": speech_db, "snr_db": speech_db - floor,
        "voiced_start": v_start * HOP, "voiced_end": v_end * HOP,
    }
    return events, info


# ── words vs. script ─────────────────────────────────────────────────────────

WER_OK = 0.10                    # a clip "says its script" at or below this word error rate
WER_DELETE = 0.25                # at/above this, don't try to rescue it
LOW_SNR_DB = 22.0                # speech less than this far above the background: too noisy


def _word_ops(script: str, heard: str):
    from faily.modules.cleanup import normalize_words
    import difflib
    a, b = normalize_words(script), normalize_words(heard)
    ops = difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes()
    lead = ops[0][4] - ops[0][3] if ops and ops[0][0] == "insert" else 0
    trail = ops[-1][4] - ops[-1][3] if len(ops) > 1 and ops[-1][0] == "insert" else 0
    inner_extra = sum(j2 - j1 for k, (op, i1, i2, j1, j2) in enumerate(ops)
                      if op in ("insert", "replace") and not (k == 0 and op == "insert") and not (k == len(ops) - 1 and op == "insert"))
    missing = sum(i2 - i1 for op, i1, i2, j1, j2 in ops if op in ("delete", "replace"))
    return a, b, lead, trail, inner_extra, missing


def _wer(a: list[str], b: list[str]) -> float:
    from faily.modules.cleanup import word_error_rate
    return word_error_rate(a, b)


def _gap_cuts(x: np.ndarray, sr: int) -> list[float]:
    """Candidate cut points (seconds): the middle of every quiet gap between sound runs."""
    f = frame_features(x, sr)
    quiet = f["excess"] <= EXCESS_DB
    cuts = [((s + e) / 2) * HOP for s, e in _runs(quiet) if (e - s) * HOP >= 0.04]
    return cuts


def _verified_edge_trim(
    x: np.ndarray, sr: int, script: str, transcribe: Callable[[np.ndarray, int], str],
    trim_lead: bool, trim_trail: bool,
) -> tuple[float, float] | None:
    """Find (start, end) seconds that drop the extra words at the edges, checking each
    candidate by re-transcribing the trimmed audio. None if no candidate says the script."""
    from faily.modules.cleanup import normalize_words
    ref = normalize_words(script)
    dur = len(x) / sr
    cuts = _gap_cuts(x, sr)
    start, end = 0.0, dur
    ok = lambda seg: _wer(ref, normalize_words(transcribe(seg, sr))) <= WER_OK
    if trim_trail:
        for c in sorted((c for c in cuts if c > dur * 0.3), reverse=True)[:6]:
            if ok(x[: int(c * sr)]):
                end = c
                break
        else:
            return None
    if trim_lead:
        base = x[: int(end * sr)]
        for c in [c for c in cuts if c < end * 0.7][:6]:
            if ok(base[int(c * sr):]):
                start = c
                break
        else:
            return None
    return start, end


def audit_clip(path: Path, script: str, transcribe: Callable[[np.ndarray, int], str]) -> dict | None:
    """Everything wrong with one generated clip, or None if it is clean.

    Returns {"issues", "events", "mute": [(a, b)], "trim": {"trim_start","trim_end",...} | None,
    "words": {...}, "recommend_delete": bool, "reason": str, "snr_db", "duration"}."""
    x, sr = _read(path)
    dur = len(x) / sr
    events, info = find_noise_events(x, sr)
    heard = transcribe(x, sr).strip()
    a, b, lead, trail, inner, missing = _word_ops(script, heard)
    wer = _wer(a, b)
    issues: list[str] = []
    out: dict = {"events": events, "duration": round(dur, 2), "snr_db": round(info["snr_db"], 1),
                 "mute": [], "trim": None, "words": None, "recommend_delete": False, "reason": ""}

    # ── words vs script ──
    t_start, t_end = 0.0, dur
    if wer > WER_OK:
        edge_only = (lead or trail) and not inner and not missing
        fixed = None
        if edge_only:
            fixed = _verified_edge_trim(x, sr, script, transcribe, bool(lead), bool(trail))
        if fixed:
            t_start, t_end = fixed
            extra = f"{lead + trail} extra word{'s' if lead + trail != 1 else ''} at the " + (
                "start and end" if lead and trail else "start" if lead else "end")
            issues.append(extra)
            out["words"] = {"heard": heard, "kind": "edge", "verified": True, "extra": lead + trail}
        else:
            n_diff = max(inner + lead + trail, missing, 1)
            issues.append(f"{n_diff} word{'s' if n_diff != 1 else ''} differ{'' if n_diff != 1 else 's'} from its script")
            out["words"] = {"heard": heard, "kind": "mismatch", "verified": False, "wer": round(wer, 2),
                            "extra": inner + lead + trail, "missing": missing}
            if wer >= WER_DELETE or missing >= 2:
                out["recommend_delete"] = True
                out["reason"] = "it doesn't say what its script says"

    # ── noise events ──
    inside = [e for e in events if e.where == "inside"]
    start_ev = [e for e in events if e.where == "start"]
    end_ev = [e for e in events if e.where == "end"]
    vs, ve = info["voiced_start"], info["voiced_end"]
    if start_ev:
        gap = max(vs - start_ev[-1].end, 0.0)
        t_start = max(t_start, start_ev[-1].end + min(0.1, gap / 2))
    if end_ev:
        gap = max(end_ev[0].start - ve, 0.0)
        t_end = min(t_end, end_ev[0].start - min(0.1, gap / 2))
    if t_start > 0.005 or t_end < dur - 0.005:
        out["trim"] = {"trim_start": round(t_start, 2), "trim_end": round(t_end, 2), "duration": round(dur, 2),
                       "speech_start": round(vs, 2), "speech_end": round(ve, 2)}
    if start_ev or end_ev:
        issues.append(f"{len(start_ev) + len(end_ev)} noise{'s' if len(start_ev) + len(end_ev) != 1 else ''} at the edges")
    if inside:
        out["mute"] = [(round(e.start, 2), round(e.end, 2)) for e in inside]
        issues.append(f"{len(inside)} noise{'s' if len(inside) != 1 else ''} between words ("
                      + ", ".join(sorted({e.kind for e in inside})) + ")")

    # ── background ──
    if info["snr_db"] < LOW_SNR_DB:
        issues.append(f"noisy background (speech only {info['snr_db']:.0f} dB above it)")
        out["recommend_delete"] = True
        out["reason"] = out["reason"] or "its background noise can't be removed"

    if not issues:
        return None
    out["issues"] = issues
    return out


def audit_character(
    name: str, progress: Callable[[int, int, str], None] | None = None, cancel: list[bool] | None = None,
) -> dict[str, dict]:
    """Audit every padding clip of a character. {clip_key: audit} for clips that need
    attention (including duplicates of an earlier clip's script). Blocking."""
    from faily.modules import cleanup as K
    from faily.modules.asr import transcribe

    clips = [c for c in K.own_clips(name) if c.source == "buffer"]
    result: dict[str, dict] = {}
    seen: dict[str, str] = {}
    for i, c in enumerate(clips):
        if cancel and cancel[0]:
            break
        if progress:
            progress(i, len(clips), c.audio.stem)
        key = " ".join(K.normalize_words(c.transcript))
        dup_of = seen.get(key)
        seen.setdefault(key, c.audio.stem)
        try:
            r = audit_clip(c.audio, c.transcript, transcribe)
        except Exception as exc:
            r = {"issues": [f"couldn't audit: {exc}"], "events": [], "mute": [], "trim": None, "words": None,
                 "recommend_delete": False, "reason": "", "duration": 0, "snr_db": 0, "error": True}
        if dup_of:
            r = r or {"events": [], "mute": [], "trim": None, "words": None, "recommend_delete": False,
                      "reason": "", "duration": 0, "snr_db": 0, "issues": []}
            r["issues"] = [f"duplicate of {dup_of}"] + r["issues"]
            r["recommend_delete"], r["reason"] = True, f"same script as {dup_of}"
        if r:
            result[c.key] = r
    return result
