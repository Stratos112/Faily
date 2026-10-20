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
EFFECT_WER = 0.05                # a noise fix may change at most this share of the words


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
    """Find (start, end) seconds that drop the extra words at the edges. Candidate cuts are
    the quiet gaps; each is checked by transcribing what it would keep, and the answer
    steers a bisection (more words than the script → cut too late/early, fewer → the
    other way), so a long stray passage with many pauses still takes ~5 attempts.
    None if no gap leaves exactly the script."""
    from faily.modules.cleanup import normalize_words
    ref = normalize_words(script)
    dur = len(x) / sr
    cuts = sorted(c for c in _gap_cuts(x, sr) if 0.3 < c < dur - 0.3)

    def words(seg):
        return normalize_words(transcribe(seg, sr))

    end = dur
    if trim_trail:
        lo, hi, found = 0, len(cuts) - 1, None
        for _ in range(10):
            if lo > hi:
                break
            mid = (lo + hi) // 2
            hyp = words(x[: int(cuts[mid] * sr)])
            if _wer(ref, hyp) <= WER_OK:
                found = cuts[mid]
                break
            if len(hyp) > len(ref):
                hi = mid - 1                 # still includes extra words: cut earlier
            else:
                lo = mid + 1                 # lost part of the script: cut later
        if found is None:
            return None
        end = found
    start = 0.0
    if trim_lead:
        head = x[: int(end * sr)]
        lcuts = [c for c in cuts if c < end - 0.3]
        lo, hi, found = 0, len(lcuts) - 1, None
        for _ in range(10):
            if lo > hi:
                break
            mid = (lo + hi) // 2
            hyp = words(head[int(lcuts[mid] * sr):])
            if _wer(ref, hyp) <= WER_OK:
                found = lcuts[mid]
                break
            if len(hyp) > len(ref):
                lo = mid + 1                 # still includes extra words: cut later
            else:
                hi = mid - 1                 # cut into the script: cut earlier
        if found is None:
            return None
        start = found
    return start, end


_NOT_SPEECH = {"", "you", "thank you", "thanks", "thank you for watching", "bye", "so"}


def _verified(x: np.ndarray, sr: int, events: list[Event], transcribe) -> list[Event]:
    """Keep only events Whisper hears no words in. Breathy, creaky or whispered speech has
    no pitch either, so the signal rule alone mistakes real words for noise — about half
    of what it flags in human recordings. On near-silence Whisper makes up "Thank you."
    or "you"; anything more is treated as speech and left alone. Clicks are too short
    to be words and need no check."""
    import re
    keep = []
    for e in events:
        if e.kind != "click":
            a, b = int(max(e.start - 0.05, 0) * sr), int(min(e.end + 0.05, len(x) / sr) * sr)
            heard = re.sub(r"[^a-z' ]", "", transcribe(x[a:b], sr).lower()).strip()
            if heard not in _NOT_SPEECH:
                continue
        keep.append(e)
    return keep


def audit_clip(
    path: Path, script: str, transcribe: Callable[[np.ndarray, int], str], generated: bool = True,
) -> dict | None:
    """Everything wrong with one clip's audio, or None if it is clean.

    generated=True (a padding clip, whose text is the script it was made from): also
    compares the words with the script, and flags a too-noisy background.
    generated=False (a human recording): noise events only — its transcript is checked
    by the normal transcript check, and a noisy room is normal.

    Returns {"issues", "edge_issues", "events", "mute": [(a, b)], "trim": {...} | None,
    "words": {...} | None, "recommend_delete": bool, "reason": str, "snr_db", "duration"}."""
    x, sr = _read(path)
    dur = len(x) / sr
    events, info = find_noise_events(x, sr)
    events = _verified(x, sr, events, transcribe)
    if generated:
        heard = transcribe(x, sr).strip()
        a, b, lead, trail, inner, missing = _word_ops(script, heard)
        wer = _wer(a, b)
    else:
        heard, a, b, lead, trail, inner, missing, wer = "", [], [], 0, 0, 0, 0, 0.0
    issues: list[str] = []            # shown as their own findings
    edge_issues: list[str] = []       # folded into the TRIM section (extra words / noise at the edges)
    out: dict = {"events": events, "duration": round(dur, 2), "snr_db": round(info["snr_db"], 1),
                 "mute": [], "trim": None, "words": None, "recommend_delete": False, "reason": ""}

    # ── words vs script ──
    t_start, t_end = 0.0, dur
    if generated and wer > WER_OK:
        edge_only = (lead or trail) and not inner and not missing
        fixed = None
        if edge_only:
            fixed = _verified_edge_trim(x, sr, script, transcribe, bool(lead), bool(trail))
        if fixed:
            t_start, t_end = fixed
            extra = f"{lead + trail} extra word{'s' if lead + trail != 1 else ''} at the " + (
                "start and end" if lead and trail else "start" if lead else "end")
            edge_issues.append(extra)
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
    # A fix is only offered if the clip still says exactly the same thing afterwards: the
    # repaired audio is re-transcribed and compared with the untouched audio. (A stretch
    # that merely sounds like noise can be breathy speech; checking just that stretch is
    # too weak, and muting it would cut a word.)
    from faily.modules.cleanup import normalize_words, process_audio
    start_ev = [e for e in events if e.where == "start"]
    end_ev = [e for e in events if e.where == "end"]
    vs, ve = info["voiced_start"], info["voiced_end"]
    ws, we = t_start, t_end                              # window left after the extra-words trim
    cs, ce = ws, we
    if start_ev:
        gap = max(vs - start_ev[-1].end, 0.0)
        cs = max(ws, start_ev[-1].end + min(0.1, gap / 2))
    if end_ev:
        gap = max(end_ev[0].start - ve, 0.0)
        ce = min(we, end_ev[0].start - min(0.1, gap / 2))
    cand = [e for e in events if e.where == "inside" and e.start >= ws and e.end <= we]
    wants_cut = cs > ws + 0.005 or ce < we - 0.005
    skipped = 0
    if cand or wants_cut:
        base = normalize_words(transcribe(x[int(ws * sr):int(we * sr)], sr))

        def same(muted, a, b_) -> bool:
            y = process_audio(x, sr, 0.0, a, b_, tuple((e.start, e.end) for e in muted))
            return _wer(base, normalize_words(transcribe(y, sr))) <= EFFECT_WER

        if same(cand, cs, ce):
            kept, cut = cand, (cs, ce)
        else:
            kept = [e for e in cand if same([e], ws, we)]
            cut = (cs, ce) if (wants_cut and same([], cs, ce)) else (ws, we)
            if kept and not same(kept, *cut):
                kept = []
            skipped = (len(cand) - len(kept)) + (len(start_ev) + len(end_ev) if wants_cut and cut == (ws, we) else 0)
        t_start, t_end = cut
    else:
        kept = []
    if t_start > 0.005 or t_end < dur - 0.005:
        out["trim"] = {"trim_start": round(t_start, 2), "trim_end": round(t_end, 2), "duration": round(dur, 2),
                       "speech_start": round(vs, 2), "speech_end": round(ve, 2)}
    cut_by_noise = (t_start > ws + 0.005 or t_end < we - 0.005)
    if cut_by_noise:
        n_edge = len(start_ev) + len(end_ev)
        edge_issues.append(f"{n_edge} noise{'s' if n_edge != 1 else ''} at the edges")
    if kept:
        out["mute"] = [(round(e.start, 2), round(e.end, 2)) for e in kept]
        issues.append(f"{len(kept)} noise{'s' if len(kept) != 1 else ''} between words ("
                      + ", ".join(sorted({e.kind for e in kept})) + ")")
    out["events"] = [e for e in events if e.where != "inside" or e in kept]
    out["skipped"] = skipped            # candidates dropped because removing them would change the words

    # ── background ──
    if generated and info["snr_db"] < LOW_SNR_DB:
        issues.append(f"noisy background (speech only {info['snr_db']:.0f} dB above it)")
        out["recommend_delete"] = True
        out["reason"] = out["reason"] or "its background noise can't be removed"

    if not issues and not edge_issues:
        return None
    out["issues"] = issues
    out["edge_issues"] = edge_issues
    return out


def audit_character(
    name: str, progress: Callable[[int, int, str], None] | None = None, cancel: list[bool] | None = None,
) -> dict[str, dict]:
    """Audit every one of a character's own clips: noise events for all of them, plus the
    script comparison, duplicates and background check for generated (padding) clips.
    {clip_key: audit} for clips that need attention. Blocking."""
    from faily.modules import cleanup as K
    from faily.modules.asr import transcribe

    clips = K.own_clips(name)
    result: dict[str, dict] = {}
    seen: dict[str, str] = {}
    for i, c in enumerate(clips):
        if cancel and cancel[0]:
            break
        if progress:
            progress(i, len(clips), c.audio.stem)
        generated = c.source == "buffer"
        dup_of = None
        if generated:                              # repeated scripts matter for generated clips only
            key = " ".join(K.normalize_words(c.transcript))
            dup_of = seen.get(key)
            seen.setdefault(key, c.audio.stem)
        try:
            r = audit_clip(c.audio, c.transcript, transcribe, generated)
        except Exception as exc:
            r = {"issues": [f"couldn't audit: {exc}"], "edge_issues": [], "events": [], "mute": [], "trim": None,
                 "words": None, "recommend_delete": False, "reason": "", "duration": 0, "snr_db": 0, "error": True}
        if dup_of:
            r = r or {"events": [], "mute": [], "trim": None, "words": None, "recommend_delete": False,
                      "reason": "", "duration": 0, "snr_db": 0, "issues": [], "edge_issues": []}
            r["issues"] = [f"duplicate of {dup_of}"] + r["issues"]
            r["recommend_delete"], r["reason"] = True, f"same script as {dup_of}"
        if r:
            result[c.key] = r
    return result
