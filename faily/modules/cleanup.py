"""Reference-clip cleanup: find clips with volume / clipping problems, stray
silence or noise at the ends, or transcripts that disagree with the audio, and
apply fixes. Pure logic — the UI lives in faily/ui/tabs/cleanup_dialog.py.

Three independent analyses (volume, trim, transcript), each returning a
suggestion per clip; plus apply helpers that edit the character's own clips in
place. Originals are copied to <character>/refs_original/ before their first
modification and deleted clips go to <character>/refs_trash/, so nothing here
is unrecoverable.
"""
from __future__ import annotations

import difflib
import json
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf

from faily.core import characters as C

PRIMARY = "__primary__"          # key for a character's ref_audio (no ref_clips file key)

# ── tunables ─────────────────────────────────────────────────────────────────
TARGET_RANGE = (-26.0, -18.0)    # auto target = the set's median speech level, kept in this range
GAIN_FLAG_DB = 5.0               # flag clips this far from the target
MAX_GAIN_DB = 12.0               # never boost / cut by more than this
PEAK_CEILING = 0.9               # gain is capped so the peak stays under this
QUIET_DB = -38.0                 # "very quiet" speech level
CLIP_LEVEL = 0.99                # samples at/above this count as clipped
CLIP_MIN_SAMPLES = 3

LEAD_PAD = 0.20                  # seconds of lead-in kept before the first speech
TAIL_PAD = 0.30                  # seconds of tail kept after the last speech
LEAD_FLAG = 0.50                 # flag if the lead-in is longer than this
TAIL_FLAG = 0.80
BLIP_MAX = 0.30                  # an edge sound shorter than this, ...
BLIP_GAP = 0.40                  # ... separated by this much silence, ...
BLIP_QUIETER_DB = 10.0           # ... and this much quieter than the speech, is noise

WER_FLAG = 0.20                  # transcript pass: flag at/above this word error rate


# ── clip enumeration ─────────────────────────────────────────────────────────

@dataclass
class Clip:
    key: str                      # file_key, or PRIMARY
    audio: Path
    transcript: str
    source: str = ""              # "buffer" for generated padding


def own_clips(name: str) -> list[Clip]:
    """The character's own, non-excluded reference clips (inherited ones belong
    to — and are cleaned on — their parent)."""
    out = []
    for e in C.get_ref_chain(name):
        if e["owner"] != name:
            continue
        out.append(Clip(e["file_key"] or PRIMARY, e["audio"], e["transcript"], e.get("source", "")))
    return out


def _read(path: Path) -> tuple[np.ndarray, int]:
    data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data, sr


def _frame_db(x: np.ndarray, sr: int, win: float = 0.02, hop: float = 0.01) -> np.ndarray:
    w, h = max(int(win * sr), 1), max(int(hop * sr), 1)
    if len(x) < w:
        x = np.pad(x, (0, w - len(x)))
    n = 1 + (len(x) - w) // h
    idx = np.arange(w)[None, :] + h * np.arange(n)[:, None]
    rms = np.sqrt(np.mean(x[idx] ** 2, axis=1))
    return 20 * np.log10(rms + 1e-9)


def speech_rms_db(x: np.ndarray, sr: int) -> float:
    """RMS (dBFS) over the speech-active frames only, so a clip padded with
    silence isn't judged quieter than it is."""
    db = _frame_db(x, sr)
    active = db > (db.max() - 30)
    w, h = max(int(0.02 * sr), 1), max(int(0.01 * sr), 1)
    vals = [x[i * h:i * h + w] for i in np.flatnonzero(active)]
    if not vals:
        return -90.0
    return float(20 * np.log10(np.sqrt(np.mean(np.concatenate(vals) ** 2)) + 1e-9))


# ── pass 1: volume / clipping ────────────────────────────────────────────────

def analyze_volume(path: Path, target_db: float) -> dict | None:
    x, sr = _read(path)
    if len(x) == 0:
        return None
    peak = float(np.max(np.abs(x)))
    level = speech_rms_db(x, sr)
    n_clipped = int(np.sum(np.abs(x) >= CLIP_LEVEL))
    issues = []
    if n_clipped >= CLIP_MIN_SAMPLES:
        issues.append(f"clipping ({n_clipped} samples at full scale)")
    if level < QUIET_DB:
        issues.append(f"very quiet ({level:.0f} dBFS)")
    gain = target_db - level
    if abs(gain) >= GAIN_FLAG_DB and not issues:
        issues.append(("too loud" if gain < 0 else "too quiet") + f" ({level:.0f} dBFS vs {target_db:.0f} target)")
    if not issues:
        return None
    gain = float(np.clip(gain, -MAX_GAIN_DB, MAX_GAIN_DB))
    if peak > 0:                                   # never push the peak past the ceiling
        gain = min(gain, 20 * np.log10(PEAK_CEILING / peak))
    # A "too quiet" clip whose peaks already reach the ceiling can't be boosted;
    # don't suggest a no-op fix.
    if abs(gain) < 1.0 and not any(i.startswith(("clipping", "very quiet")) for i in issues):
        return None
    return {
        "issues": issues, "gain_db": round(gain, 1), "level_db": round(level, 1),
        "peak": round(peak, 3), "target_db": round(target_db, 1),
        "off_db": round(level - target_db, 1),            # >0: louder than the set's typical level
        "note": ("Lowering the volume evens it out but can't restore the flattened peaks."
                 if n_clipped >= CLIP_MIN_SAMPLES else ""),
    }


# ── pass 2: leading / trailing silence and edge noise ────────────────────────

def analyze_trim(path: Path, transcript: str = "") -> dict | None:
    x, sr = _read(path)
    dur = len(x) / sr if sr else 0
    if dur < 0.6:
        return None
    db = _frame_db(x, sr)
    hop = 0.01
    floor = float(np.percentile(db, 10))
    top = float(np.percentile(db, 95))
    thr = max(floor + 10.0, top - 30.0)
    active = db > thr
    # runs of active frames -> [start_s, end_s, mean_db]
    segs, i, n = [], 0, len(active)
    while i < n:
        if active[i]:
            j = i
            while j + 1 < n and active[j + 1]:
                j += 1
            segs.append([i * hop, (j + 1) * hop + 0.02, float(db[i:j + 1].mean())])
            i = j + 1
        else:
            i += 1
    if not segs:
        return None
    merged = [segs[0]]
    for s_, e_, m_ in segs[1:]:                  # bridge short between-word gaps
        if s_ - merged[-1][1] < 0.25:
            w0, w1 = merged[-1][1] - merged[-1][0], e_ - s_
            merged[-1] = [merged[-1][0], e_, (merged[-1][2] * w0 + m_ * w1) / max(w0 + w1, 1e-6)]
        else:
            merged.append([s_, e_, m_])
    main_db = max(merged, key=lambda m: m[1] - m[0])[2]

    def is_blip(seg, neighbour, edge_first: bool) -> bool:
        gap = (neighbour[0] - seg[1]) if edge_first else (seg[0] - neighbour[1])
        return (seg[1] - seg[0] < BLIP_MAX and gap >= BLIP_GAP
                and seg[2] < main_db - BLIP_QUIETER_DB)

    reasons = []
    while len(merged) > 1 and is_blip(merged[0], merged[1], True):
        merged.pop(0); reasons.append("noise before the speech")
    while len(merged) > 1 and is_blip(merged[-1], merged[-2], False):
        merged.pop(); reasons.append("noise after the speech")
    # A transcript that starts / ends with a filler ("umm. So…") means the quiet
    # opening sound is wanted speech, not noise — keep it.
    raw = re.sub(r"[^a-z' ]+", " ", transcript.lower()).split()
    if raw and raw[0] in _FILLERS | {"mhm"} and "noise before the speech" in reasons:
        reasons.remove("noise before the speech")
        merged = [segs[0]] + merged if segs[0] not in merged else merged
    if raw and raw[-1] in _FILLERS | {"mhm"} and "noise after the speech" in reasons:
        reasons.remove("noise after the speech")
        merged = merged + [segs[-1]] if segs[-1] not in merged else merged
    onset, offset = merged[0][0], merged[-1][1]
    lead, tail = onset, dur - offset
    t_start = max(0.0, onset - LEAD_PAD) if (lead > LEAD_FLAG or "noise before the speech" in reasons) else 0.0
    t_end = min(dur, offset + TAIL_PAD) if (tail > TAIL_FLAG or "noise after the speech" in reasons) else dur
    if lead > LEAD_FLAG:
        reasons.append(f"{lead:.1f}s of silence at the start")
    if tail > TAIL_FLAG:
        reasons.append(f"{tail:.1f}s of silence at the end")
    if t_start <= 0.0 and t_end >= dur:
        return None
    return {
        "issues": sorted(set(reasons)), "duration": round(dur, 2),
        "trim_start": round(t_start, 2), "trim_end": round(t_end, 2),
        "speech_start": round(onset, 2), "speech_end": round(offset, 2),
    }


# ── pass 3: transcript vs. what Whisper hears ────────────────────────────────

_FILLERS = {"um", "umm", "uh", "uhh", "er", "erm", "hmm", "mm", "mmm", "ah", "eh", "huh"}
_EQUIV = {
    "ok": "okay", "gonna": "goingto", "wanna": "wantto", "gotta": "gotto",
    "kinda": "kindof", "sorta": "sortof", "cuz": "because", "cause": "because",
    "yeah": "yeah", "yep": "yeah", "yup": "yeah", "alright": "alright",
}


def normalize_words(text: str) -> list[str]:
    t = text.lower().replace("’", "'")
    t = re.sub(r"\b(\d+)\s*(a|p)\.?\s?m\.?\b", r"\1\2m", t)
    t = re.sub(r"(\w)-(\w)", r"\1 \2", t)
    t = re.sub(r"[^a-z0-9' ]+", " ", t)
    words = []
    for w in t.split():
        if w in _FILLERS:
            continue
        words.append(_EQUIV.get(w, w))
    # "going to" / "want to" / "got to" spoken in full == their contractions
    out, i = [], 0
    while i < len(words):
        pair = words[i] + (words[i + 1] if i + 1 < len(words) else "")
        if pair in ("goingto", "wantto", "gotto") and words[i] != pair:
            out.append(pair); i += 2
        else:
            out.append(words[i]); i += 1
    return out


def has_filler(text: str) -> bool:
    return any(w in _FILLERS | {"mhm"} for w in re.findall(r"[a-z']+", text.lower()))


def word_error_rate(ref: list[str], hyp: list[str]) -> float:
    if not ref:
        return 1.0 if hyp else 0.0
    d = np.zeros((len(ref) + 1, len(hyp) + 1), dtype=int)
    d[:, 0] = range(len(ref) + 1)
    d[0, :] = range(len(hyp) + 1)
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]))
    return float(d[-1, -1]) / len(ref)


def count_differences(current: str, suggested: str) -> int:
    """How many words differ between two transcripts (ignoring case, punctuation,
    fillers and spoken-vs-contracted forms)."""
    a, b = normalize_words(current), normalize_words(suggested)
    return sum(max(i2 - i1, j2 - j1)
               for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes()
               if op != "equal")


def diff_words(current: str, suggested: str) -> list[tuple[str, str]]:
    """Word-level diff of the suggested text against the current one:
    [(word, "same"|"added"|"changed")] over the *suggested* words, plus
    ("", "removed") markers are not emitted — the UI shows both strings."""
    a, b = current.split(), suggested.split()
    na = [normalize_words(w)[0] if normalize_words(w) else w.lower() for w in a]
    nb = [normalize_words(w)[0] if normalize_words(w) else w.lower() for w in b]
    out = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=na, b=nb, autojunk=False).get_opcodes():
        for w in b[j1:j2]:
            out.append((w, "same" if op == "equal" else ("added" if op == "insert" else "changed")))
    return out


def analyze_transcript(path: Path, transcript: str, transcribe: Callable[[np.ndarray, int], str]) -> dict | None:
    x, sr = _read(path)
    hyp = transcribe(x, sr).strip()
    ref_w, hyp_w = normalize_words(transcript), normalize_words(hyp)
    wer = word_error_rate(ref_w, hyp_w)
    if not transcript.strip():
        issues = ["no transcript"]
    elif wer >= WER_FLAG or abs(len(ref_w) - len(hyp_w)) >= 3:
        issues = [f"{wer:.0%} of the words differ from what Whisper hears"]
    else:
        return None
    return {"issues": issues, "wer": round(wer, 2), "suggested": hyp}


_HALLUCINATIONS = {"", "you", "thank you", "thanks", "thank you for watching", "bye", "so"}


def _annotate_cuts(path: Path, r: dict, transcribe) -> None:
    """Ask Whisper what is in the pieces a trim would remove. On near-silence it
    invents "Thank you." / "you"; anything else is probably real, so flag it."""
    x, sr = _read(path)
    heard = []
    for side, seg in (("start", x[:int(r["trim_start"] * sr)]),
                      ("end", x[int(r["trim_end"] * sr):] if r["trim_end"] < r["duration"] - 0.01 else x[:0])):
        if len(seg) < 0.15 * sr:
            continue
        t = transcribe(seg, sr).strip()
        if re.sub(r"[^a-z ]", "", t.lower()).strip() not in _HALLUCINATIONS:
            heard.append(f"{side}: “{t}”")
    if heard:
        r["caution"] = "Whisper hears speech in the part being cut — " + "; ".join(heard)


# ── whole-character analysis ─────────────────────────────────────────────────

def analyze_all(
    name: str, passes: tuple[str, ...] = ("volume", "trim", "transcript"),
    target_db: float | None = None,
    progress: Callable[[str, int, int, str], None] | None = None,
    cancel: list[bool] | None = None,
) -> dict[str, dict[str, dict]]:
    """{pass_name: {clip_key: suggestion}} for the clips that need attention.
    progress(pass, i, n, clip_name) is called per clip. Blocking."""
    clips = own_clips(name)
    out: dict[str, dict[str, dict]] = {p: {} for p in passes}
    if "volume" in passes and target_db is None:
        levels = []
        for c in clips:
            try:
                x, sr = _read(c.audio)
                levels.append(speech_rms_db(x, sr))
            except Exception:
                pass
        target_db = float(np.clip(np.median(levels), *TARGET_RANGE)) if levels else -23.0
    transcribe = None
    if "transcript" in passes:
        from faily.modules.asr import transcribe as _t
        transcribe = _t
    for p in passes:
        for i, clip in enumerate(clips):
            if cancel and cancel[0]:
                return out
            if progress:
                progress(p, i, len(clips), clip.audio.stem)
            try:
                if p == "volume":
                    r = analyze_volume(clip.audio, target_db)
                elif p == "trim":
                    r = analyze_trim(clip.audio, clip.transcript)
                    if r and transcribe and any("noise" in i for i in r["issues"]):
                        _annotate_cuts(clip.audio, r, transcribe)
                else:
                    r = analyze_transcript(clip.audio, clip.transcript, transcribe)
            except Exception as exc:       # one unreadable clip shouldn't sink the run
                r = {"issues": [f"couldn't analyze: {exc}"], "error": True}
            if r:
                out[p][clip.key] = r
    return out


# ── applying fixes ───────────────────────────────────────────────────────────

def _char_dir(name: str) -> Path:
    return C.CHARACTERS_DIR / name


def _backup(name: str, audio: Path) -> None:
    dest = _char_dir(name) / "refs_original" / audio.name
    if not dest.exists() and audio.exists():
        dest.parent.mkdir(exist_ok=True)
        shutil.copy2(audio, dest)


def process_audio(x: np.ndarray, sr: int, gain_db: float = 0.0,
                  trim_start: float = 0.0, trim_end: float | None = None) -> np.ndarray:
    a = int(max(trim_start, 0.0) * sr)
    b = len(x) if trim_end is None else int(min(trim_end * sr, len(x)))
    y = x[a:b] * (10 ** (gain_db / 20))
    return np.clip(y, -1.0, 1.0).astype("float32")


def render_preview(path: Path, out: Path, gain_db: float = 0.0,
                   trim_start: float = 0.0, trim_end: float | None = None) -> Path:
    x, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out), process_audio(x, sr, gain_db, trim_start, trim_end), sr)
    return out


def apply_audio_fix(name: str, clip: Clip, gain_db: float = 0.0,
                    trim_start: float = 0.0, trim_end: float | None = None) -> Clip:
    """Rewrite the clip with gain / trim applied. Non-WAV clips (e.g. mp3) are
    written as WAV and the character config is repointed at the new file.
    Returns the updated Clip."""
    if abs(gain_db) < 0.05 and trim_start <= 0 and trim_end is None:
        return clip
    x, sr = sf.read(str(clip.audio), dtype="float32", always_2d=False)
    mono = x.mean(axis=1) if x.ndim > 1 else x
    y = process_audio(mono, sr, gain_db, trim_start, trim_end)
    _backup(name, clip.audio)
    if clip.audio.suffix.lower() == ".wav":
        sf.write(str(clip.audio), y, sr, subtype="PCM_16")
        return clip
    new_audio = clip.audio.with_suffix(".wav")
    sf.write(str(new_audio), y, sr, subtype="PCM_16")
    cfg_path = _char_dir(name) / "config.json"
    cfg = json.loads(cfg_path.read_text())
    rel = new_audio.relative_to(_char_dir(name)).as_posix()
    new_key = clip.key
    if clip.key == PRIMARY:
        cfg["ref_audio"] = rel
    else:
        for rc in cfg.get("ref_clips", []):
            if rc["file"] == clip.key:
                rc["file"] = rel
        new_key = rel
    cfg_path.write_text(json.dumps(cfg, indent=2))
    clip.audio.unlink()                      # the original is in refs_original/
    return Clip(new_key, new_audio, clip.transcript, clip.source)


def delete_padding_clips(name: str) -> int:
    """Remove every generated padding ("buffer") clip from the character's own
    reference clips — excluded ones too. The audio goes to refs_trash/ (so it can be
    recovered by hand) and the config is rewritten once. Returns how many were removed."""
    cfg_path = _char_dir(name) / "config.json"
    cfg = json.loads(cfg_path.read_text())
    keep, gone = [], []
    for rc in cfg.get("ref_clips", []):
        (gone if rc.get("source") == "buffer" else keep).append(rc)
    if not gone:
        return 0
    trash = _char_dir(name) / "refs_trash"
    trash.mkdir(exist_ok=True)
    for rc in gone:
        audio = _char_dir(name) / rc["file"]
        if audio.exists():
            shutil.move(str(audio), str(trash / audio.name))
    cfg["ref_clips"] = keep
    cfg_path.write_text(json.dumps(cfg, indent=2))
    return len(gone)


def set_transcript(name: str, clip: Clip, text: str) -> None:
    text = " ".join(text.split())
    # Keep a log of what each transcript was before it was changed.
    log_path = _char_dir(name) / "refs_original" / "transcript_changes.json"
    log_path.parent.mkdir(exist_ok=True)
    try:
        log = json.loads(log_path.read_text())
    except (OSError, ValueError):
        log = []
    log.append({"clip": clip.key, "before": clip.transcript, "after": text, "at": time.strftime("%Y-%m-%d %H:%M:%S")})
    log_path.write_text(json.dumps(log, indent=2))
    if clip.key == PRIMARY:
        C.update_character_metadata(name, {"transcript": text})
    else:
        C.update_ref_clip(name, clip.key, None, text)


def delete_clip(name: str, clip: Clip) -> bool:
    """Move the clip to refs_trash/ and drop it from the character. The primary
    reference can't be removed this way (a character needs one). Returns
    whether anything was deleted."""
    if clip.key == PRIMARY:
        return False
    trash = _char_dir(name) / "refs_trash"
    trash.mkdir(exist_ok=True)
    if clip.audio.exists():
        shutil.copy2(clip.audio, trash / clip.audio.name)
    C.remove_ref_clip(name, clip.key)
    return True
