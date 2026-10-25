"""Automatic qualification of freshly inflated clips (no human stepping).

An inflated clip is zero-shot TTS output of a KNOWN script, so instead of hunting for
defects we define what the clip is and treat everything else as junk:

  A. Align — forced CTC alignment of the script (spoken form: numbers, dollars, … are
     written out) against the audio gives a start and end time for every word. The clip is
     cut to the first word's start / the last word's end, plus a small margin set by the
     script's punctuation. Whatever the model emitted after the words (a breath, a moan, a
     laugh, seconds of silence) is simply outside the cut.
  B. Leftover — audible sound that is not explained by a word is found whatever it sounds
     like (pitched or not): at the edges it is cut away by A, between words it is muted if
     short and the clip is rejected if not.

Then Whisper transcribes the result and it must say the script. A clip comes back as one of:
  ok      kept (possibly trimmed / with short gap sounds muted)
  review  the tool isn't sure (alignment doubt, a word that didn't align, Whisper disagrees
          a little): left for a person
  reject  clearly bad (sound between words, Whisper says something else)

Pure logic; model loading goes through the ModelManager like Whisper does.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf

from faily.core.model_manager import manager
from faily.modules import qualify as K

MODEL_ID = "facebook/wav2vec2-base-960h"
FRAME = 0.02                     # wav2vec2 output stride (seconds)
HOP = 0.01                       # energy-envelope hop (seconds)

# ── tunables ─────────────────────────────────────────────────────────────────
MIN_CLIP_SCORE = 0.35            # mean word alignment confidence below this -> review
MIN_WORD_SCORE = 0.08            # any single word below this -> review (didn't really align)
AUDIBLE_BELOW_SPEECH_DB = 32.0   # a frame this far under the speech level or louder is "sound"
AUDIBLE_ABOVE_FLOOR_DB = 10.0    # ...and at least this far over the clip's own noise floor
MIN_EVENT = 0.10                 # seconds of leftover sound before it counts
EVENT_GUARD = 0.03               # leftover sound this close to a word is the word's own onset/tail
MUTE_MAX = 0.35                  # a sound between words up to this long is muted; longer -> reject
LOUD_EVENT_DB = 18.0             # a gap sound within this many dB of the speech is too loud to mute
START_WALK = 0.15                # how far the first word's start may be pulled back to catch its onset
INNER_START_WALK = 0.08          # ...and a later word's onset
END_WALK = 0.40                  # how far the last word's end may be pushed out to let its sound finish
DECAY_ABOVE_FLOOR_DB = 6.0       # a tail counts as finished once it is this close to the clip's noise floor
DECAY_BELOW_SPEECH_DB = 38.0     # ...or this far under the speech level, whichever is louder
TAIL_BELOW_SPEECH_DB = 20.0      # under this (relative to speech) the sound is in its tail: rises/flats can end it
RELEASE_WITHIN = 0.15            # an unvoiced burst this soon after the word's end is its release (t, k, p)
RELEASE_RISE_DB = 10.0           # a burst must stand this far over the dip before it
RELEASE_MAX = 0.20               # ...and is followed for at most this long
NEW_SOUND_RISE_DB = 6.0          # ...unless it rises this much after dipping: that is a new sound, not the tail
INNER_END_WALK = 0.25            # a word followed by another can end in a longer "s"/"sh"
FADE_IN = 0.015                  # seconds: smooth start (inside the cushion before the first word)
FADE_OUT = 0.06                  # seconds: smooth end (inside the cushion after the last word)
EXTRA_START = 0.03              # always keep this much more before the first word's measured start
EXTRA_END = 0.12                 # ...and this much more after the last word's measured end
LEAD = 0.05                      # silence kept before the first word
TAIL = {".": 0.20, "!": 0.20, "?": 0.20, "…": 0.20, ";": 0.15, ":": 0.15, ",": 0.15}
TAIL_DEFAULT = 0.10
WER_OK = 0.10
WER_REJECT = 0.25
MAX_SECONDS = 40.0


@dataclass
class Result:
    status: str = "ok"                       # ok | review | reject
    reasons: list[str] = field(default_factory=list)
    trimmed_start: float = 0.0               # seconds removed from the front
    trimmed_end: float = 0.0                 # seconds removed from the end
    muted: int = 0
    score: float = 0.0                       # mean word alignment confidence
    wer: float | None = None
    words: list[tuple[str, float, float, float]] = field(default_factory=list)   # (word, start, end, score)

    def summary(self) -> str:
        bits = list(self.reasons)
        if self.status == "ok":
            if self.trimmed_start + self.trimmed_end > 0.05:
                bits.append(f"trimmed {self.trimmed_start:.2f} s + {self.trimmed_end:.2f} s")
            if self.muted:
                bits.append(f"muted {self.muted} gap sound{'s' if self.muted != 1 else ''}")
        return "; ".join(bits)


# ── script -> spoken words ───────────────────────────────────────────────────

_ABBR = {
    "dr": "doctor", "mr": "mister", "mrs": "missus", "ms": "miss", "vs": "versus",
    "a.m": "a m", "p.m": "p m", "u.s": "u s", "&": "and", "st": "street", "no": "no",
}
_MAGNITUDE = {"million", "billion", "thousand", "trillion"}


def _num(n: int | float, **kw) -> str:
    from num2words import num2words
    return num2words(n, **kw).replace("-", " ").replace(",", "").replace(" and ", " ")


def _digits(s: str) -> str:
    return " ".join(_num(int(c)) if c != "0" else "oh" for c in s if c.isdigit())


def _spoken_number(s: str, year_style: bool) -> str:
    s = s.replace(",", "")
    if "." in s:
        whole, frac = s.split(".", 1)
        out = _num(int(whole or 0)) + " point " + " ".join(_num(int(c)) if c != "0" else "zero" for c in frac)
        return out
    n = int(s)
    if year_style and len(s) == 4 and (1100 <= n <= 1999 or 2010 <= n <= 2099):
        return _num(n, to="year")
    return _num(n)


def spoken_tokens(script: str, year_style: bool = True) -> list[tuple[str, str]]:
    """The script as [(spoken text, trailing punctuation of the original token)]. Each entry
    is one original whitespace token written out as words ("$63.47," -> "sixty three dollars
    and forty seven cents", ","). Characters the aligner can't use are dropped later."""
    toks = script.replace("’", "'").split()
    out: list[tuple[str, str]] = []
    i = 0
    while i < len(toks):
        raw = toks[i]
        m = re.match(r"^([\"'(\[“‘]*)(.*?)([\"')\]”’.,;:!?…]*)$", raw)
        core, trail = m.group(2), m.group(3)
        punct = trail[-1:] if trail else ""
        # keep a period that belongs to an abbreviation (Dr., a.m.)
        key = core.lower().rstrip(".")
        nxt = toks[i + 1].lower().strip(".,;:!?") if i + 1 < len(toks) else ""
        words = ""
        if key in _ABBR and (trail.startswith(".") or key in ("&", "a.m", "p.m", "u.s")):
            words = _ABBR[key]
            if key in ("a.m", "p.m", "u.s") and trail.startswith("."):
                trail = trail[1:] or ""
                punct = trail[-1:] if trail else ""
        elif re.fullmatch(r"\$[\d,]+(\.\d+)?", core):
            amt = core[1:].replace(",", "")
            if nxt in _MAGNITUDE:
                words = f"{_spoken_number(amt, False)} {nxt} dollars"
                i += 1
                trail = toks[i].lstrip("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
                punct = trail[-1:] if trail else ""
            elif "." in amt:
                d, c = amt.split(".", 1)
                c = (c + "0")[:2]
                words = f"{_num(int(d or 0))} dollar{'s' if int(d or 0) != 1 else ''}"
                if int(c):
                    words += f" and {_num(int(c))} cent{'s' if int(c) != 1 else ''}"
            else:
                words = f"{_num(int(amt))} dollar{'s' if int(amt) != 1 else ''}"
        elif re.fullmatch(r"-?[\d,]+(\.\d+)?%", core):
            words = ("minus " if core.startswith("-") else "") + _spoken_number(core.strip("-%"), False) + " percent"
        elif re.fullmatch(r"\d{1,2}:\d{2}", core):
            h, mi = core.split(":")
            words = _num(int(h)) if mi == "00" and nxt not in ("a.m", "p.m", "am", "pm") else _num(int(h))
            if mi != "00":
                words += (" oh " if int(mi) < 10 else " ") + _num(int(mi))
            elif nxt not in ("a.m", "p.m", "am", "pm"):
                words += " o'clock"
        elif re.fullmatch(r"\d+(st|nd|rd|th)", core, re.I):
            words = _num(int(re.match(r"\d+", core).group()), to="ordinal")
        elif re.fullmatch(r"\d+/\d+", core):
            a, b = core.split("/")
            den = {"2": "half", "3": "third", "4": "quarter"}.get(b) or _num(int(b), to="ordinal")
            words = f"{_num(int(a))} {den}" + ("s" if int(a) > 1 and b != "2" else "") if not (a == "1" and b == "2") else "one half"
            if int(a) > 1 and b == "2":
                words = f"{_num(int(a))} halves"
        elif re.fullmatch(r"\d{2,}(-\d+)+", core):                        # 555-0142, 4-7-1-9
            parts = core.split("-")
            words = " ".join(_digits(p) if len(parts) > 2 or len(p) > 2 else _num(int(p)) for p in parts)
        elif re.fullmatch(r"-?[\d,]+(\.\d+)?", core):
            neg = core.startswith("-")
            words = ("minus " if neg else "") + _spoken_number(core.lstrip("-"), year_style and not neg)
        else:
            parts = re.split(r"(?<=\D)-(?=\D)|-(?=\d)|(?<=\d)-(?=\D)", core) if "-" in core else [core]
            sub = []
            for p in parts:
                if re.fullmatch(r"\d+", p):
                    sub.append(_num(int(p)))
                else:
                    sub.append(p)
            words = " ".join(sub)
        out.append((words, punct))
        i += 1
    return out


def _aligner_chars(words: str, vocab: dict[str, int]) -> list[str]:
    cleaned = re.sub(r"[^A-Za-z' ]+", " ", words).upper()
    return [w for w in cleaned.split() if any(c in vocab for c in w)]


# ── CTC forced alignment ─────────────────────────────────────────────────────

def _load_aligner():
    import torch
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

    def load(local_only: bool):
        return (Wav2Vec2Processor.from_pretrained(MODEL_ID, local_files_only=local_only),
                Wav2Vec2ForCTC.from_pretrained(MODEL_ID, local_files_only=local_only))
    try:
        proc, model = load(True)
    except OSError:
        proc, model = load(False)
    return proc, model.to(manager.device).eval()


def unload() -> None:
    manager.unload("ctc_aligner")


def _emissions(x16: np.ndarray) -> np.ndarray:
    import torch
    proc, model = manager.load("ctc_aligner", _load_aligner)
    inp = proc(x16, sampling_rate=16000, return_tensors="pt").input_values.to(manager.device)
    with torch.no_grad():
        logits = model(inp).logits[0].float()
    return torch.log_softmax(logits, dim=-1).cpu().numpy()          # [T, vocab]


def _viterbi(logp: np.ndarray, labels: list[int], blank: int) -> np.ndarray | None:
    """Best CTC path through `labels`. Returns, per frame, the index (into labels) being
    emitted or -1 for blank; None if the audio is too short for the text."""
    T, L = logp.shape[0], len(labels)
    S = 2 * L + 1
    ext = np.full(S, blank, dtype=int)
    ext[1::2] = labels
    NEG = -1e30
    dp = np.full((T, S), NEG)
    bp = np.zeros((T, S), dtype=np.int8)
    dp[0, 0] = logp[0, blank]
    if S > 1:
        dp[0, 1] = logp[0, ext[1]]
    can_skip = np.zeros(S, dtype=bool)
    can_skip[3::2] = ext[3::2] != ext[1:-2:2]            # odd states whose label differs from the previous label
    for t in range(1, T):
        prev = dp[t - 1]
        stay = prev
        step = np.concatenate([[NEG], prev[:-1]])
        skip = np.concatenate([[NEG, NEG], prev[:-2]])
        skip = np.where(can_skip, skip, NEG)
        stack = np.stack([stay, step, skip])
        choice = stack.argmax(axis=0)
        dp[t] = stack[choice, np.arange(S)] + logp[t, ext]
        bp[t] = choice
    end = S - 1 if dp[-1, S - 1] >= dp[-1, S - 2] else S - 2
    if dp[-1, end] <= NEG / 2:
        return None
    path = np.empty(T, dtype=int)
    s = end
    for t in range(T - 1, -1, -1):
        path[t] = s
        s -= int(bp[t, s])
    out = np.full(T, -1, dtype=int)
    odd = path % 2 == 1
    out[odd] = (path[odd] - 1) // 2
    return out


def align_words(x16: np.ndarray, spoken: list[str]) -> tuple[list[tuple[int, float, float, float]], float] | None:
    """Align `spoken` words (upper-case letters) to 16 kHz audio. Returns
    ([(word_index, start_s, end_s, score)], mean_score) or None if it can't be done."""
    proc, _ = manager.load("ctc_aligner", _load_aligner)
    vocab = proc.tokenizer.get_vocab()
    blank, sep = vocab["<pad>"], vocab["|"]
    labels, owner = [], []                  # token ids and the word each belongs to (-1 = separator)
    for wi, w in enumerate(spoken):
        if wi:
            labels.append(sep); owner.append(-1)
        for c in w:
            if c in vocab:
                labels.append(vocab[c]); owner.append(wi)
    if not labels:
        return None
    logp = _emissions(x16)
    path = _viterbi(logp, labels, blank)
    if path is None:
        return None
    res = []
    frames_of: dict[int, list[int]] = {}
    for t, li in enumerate(path):
        if li >= 0 and owner[li] >= 0:
            frames_of.setdefault(owner[li], []).append(t)
    scores = []
    for wi in range(len(spoken)):
        fr = frames_of.get(wi)
        if not fr:
            res.append((wi, 0.0, 0.0, 0.0)); scores.append(0.0)
            continue
        probs = [float(np.exp(logp[t, labels[path[t]]])) for t in fr]
        sc = float(np.mean(probs))
        res.append((wi, fr[0] * FRAME, (fr[-1] + 1) * FRAME, sc))
        scores.append(sc)
    return res, float(np.mean(scores))


# ── the per-clip procedure ───────────────────────────────────────────────────

def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    out, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append((start, i)); start = None
    if start is not None:
        out.append((start, len(mask)))
    return out


def _find_release(db: np.ndarray, voiced: np.ndarray, j: int, start: int) -> int | None:
    """A word can end with a closure and then a release (the t in "cat", a final k or p): the
    level dips, then an unvoiced burst follows. If one begins within RELEASE_WITHIN of the
    word's end (`start`), return its first frame."""
    last = min(start + int(RELEASE_WITHIN / HOP), len(db) - 1)
    low = float(db[j]) if j < len(db) else 0.0
    for k in range(j, last + 1):
        low = min(low, float(db[k]))
        if db[k] > low + RELEASE_RISE_DB and k + 4 <= len(voiced) and float(np.mean(voiced[k:k + 4])) < 0.5:
            return k
    return None


def _walk(db: np.ndarray, start: int, step: int, cap: int, soft: float, limit: int, speech_db: float,
          voiced: np.ndarray | None = None) -> int:
    """Move `start` outward (step -1 = earlier, +1 = later) until the sound that touches it is
    over. While the level is still near the speech level (a held vowel, an "s"/"sh") it keeps
    going whatever the wiggle in level; once it has fallen into the tail it stops when it
    reaches `soft` (the noise floor), flattens out (steady room tone), rises again by
    NEW_SOUND_RISE_DB over the quietest point (a new sound: a breath, a moan), or runs out of
    `cap` frames / hits `limit`. The exception: an unvoiced burst soon after the word's end is
    the word's own release (see _find_release) and is followed for RELEASE_MAX seconds."""
    i = start
    release_until, release_done = -1, voiced is None or step < 0
    tail_low = None                         # quietest level seen since the tail began
    recent: list[float] = []

    def stop_or_release(j: int) -> bool:
        """True = really stop here. Otherwise jumps into a release burst."""
        nonlocal i, release_until, release_done, tail_low, recent
        if not release_done:
            k = _find_release(db, voiced, j, start)
            if k is not None:
                release_done = True
                release_until = k + int(RELEASE_MAX / HOP)
                i, tail_low, recent = k, None, []
                return False
        return True

    for _ in range(cap):
        j = i if step > 0 else i - 1
        if j < 0 or j >= len(db) or (step > 0 and j >= limit) or (step < 0 and j < limit):
            break
        v = float(db[j])
        if j < release_until:                # inside an allowed release burst: just follow it
            i = i + 1 if step > 0 else i - 1
            continue
        if v <= soft:
            if stop_or_release(j):
                break
            continue
        if v < speech_db - TAIL_BELOW_SPEECH_DB:
            if tail_low is not None and v > tail_low + NEW_SOUND_RISE_DB:
                if stop_or_release(j):
                    break
                continue
            tail_low = v if tail_low is None else min(tail_low, v)
            recent.append(v)
            if len(recent) >= 6 and max(recent[-6:]) - min(recent[-6:]) < 2.0:
                break                        # flat: that's the room, not the word
        i = i + 1 if step > 0 else i - 1
    return i


def _tail_for(script: str) -> float:
    t = script.rstrip().rstrip("\"'”’)]»")
    return TAIL.get(t[-1:], TAIL_DEFAULT)


def _align_best(x16: np.ndarray, script: str):
    """Try the year-style and cardinal readings of numbers; keep the better alignment."""
    variants = [True] if not re.search(r"\b\d{4}\b", script) else [True, False]
    best = None
    for ys in variants:
        toks = spoken_tokens(script, ys)
        aligner_words, owner = [], []          # one entry per aligned word; owner = original token
        proc, _ = manager.load("ctc_aligner", _load_aligner)
        vocab = proc.tokenizer.get_vocab()
        for ti, (spoken, _) in enumerate(toks):
            for w in _aligner_chars(spoken, vocab):
                aligner_words.append(w); owner.append(ti)
        if not aligner_words:
            continue
        got = align_words(x16, aligner_words)
        if got and (best is None or got[1] > best[0][1]):
            best = (got, aligner_words, owner, toks)
    return best


@dataclass
class Analysis:
    """What the aligner and the signal say about one clip — everything needed to cut it, with
    nothing written. qualify_clip() applies it; the manual qualify screen shows it, so the two
    always agree."""
    status: str = "ok"                       # ok | review | reject
    reasons: list[str] = field(default_factory=list)
    score: float = 0.0
    duration: float = 0.0
    speech_start: float = 0.0                # measured edges of the words (before the cushion)
    speech_end: float = 0.0
    cut: tuple[float, float] | None = None   # (start, end) seconds to keep
    mute: list[tuple[float, float]] = field(default_factory=list)       # sounds between words
    events: list[tuple[float, float, str, float]] = field(default_factory=list)   # (start, end, where, dB vs speech)
    words: list[tuple[str, float, float, float]] = field(default_factory=list)
    aligned: bool = False                    # False = couldn't align (no cut / events available)


def analyze_clip(path: Path, script: str, recording: bool = False) -> Analysis:
    """Align `script` to the clip and work out the cut, the sounds to mute, and the verdict.

    recording=True is for a person's recording: a sound between words is as likely a natural
    breath or laugh as junk, and it is not ours to mute, so it is flagged for a look instead."""
    an = Analysis()
    x, sr = sf.read(str(path), dtype="float32", always_2d=False)
    mono = x.mean(axis=1) if x.ndim > 1 else x
    an.duration = len(mono) / sr
    if an.duration > MAX_SECONDS or len(mono) < sr * 0.2:
        an.status, an.reasons = "review", ["clip length is unusual"]
        return an

    x16 = K_resample(mono, sr)
    best = _align_best(x16, script)
    if best is None:
        an.status, an.reasons = "review", ["couldn't align the script to the audio"]
        return an
    (aligned, mean_score), aligner_words, owner, toks = best
    an.score = mean_score
    an.words = [(aligner_words[wi], s, e, sc) for wi, s, e, sc in aligned]
    if mean_score < MIN_CLIP_SCORE:
        an.status, an.reasons = "review", [f"words don't match the audio well (alignment {mean_score:.2f})"]
        return an
    worst = min(aligned, key=lambda a: a[3])
    if worst[3] < MIN_WORD_SCORE:
        an.status, an.reasons = "review", [f"“{aligner_words[worst[0]].lower()}” isn't clearly in the audio"]
        return an

    # ── energy envelope + the words' (refined) spans ─────────────────────────
    db = K._frame_db(mono, sr)                      # 20 ms windows, 10 ms hop
    n = len(db)
    to_f = lambda t: int(round(t / HOP))
    spans = [(to_f(st), max(to_f(en), to_f(st) + 1)) for _, st, en, _ in aligned]
    in_span = np.zeros(n, dtype=bool)
    for a, b in spans:
        in_span[max(a, 0):min(b, n)] = True
    speech_db = float(np.percentile(db[in_span], 75)) if in_span.any() else float(db.max())
    floor_db = float(np.percentile(db, 5))
    thresh = max(floor_db + AUDIBLE_ABOVE_FLOOR_DB, speech_db - AUDIBLE_BELOW_SPEECH_DB)
    audible = db > thresh

    # Each word's span is a spike per letter, so it misses a trailing "s"/"sh" or a soft onset.
    # Grow every span over the audible frames touching it, more between words than at the
    # clip's outer edges, where a fresh rise in energy is a new sound (a breath, a moan).
    from faily.modules.inflation_audit import frame_features
    harm = frame_features(mono, sr)["harm"]
    voiced = np.zeros(n, dtype=bool)
    voiced[:min(len(harm), n)] = harm[:n] > 0.5
    soft = max(floor_db + DECAY_ABOVE_FLOOR_DB, speech_db - DECAY_BELOW_SPEECH_DB)   # a word's tail is over once it is this quiet
    last_k = len(spans) - 1
    for k in range(len(spans)):
        s0, e0 = spans[k]
        lo = spans[k - 1][1] if k else 0
        hi = spans[k + 1][0] if k < last_k else n
        if k == 0:                                   # outer start: the onset, until the level is down at the floor
            s0 = _walk(db, s0, -1, int(START_WALK / HOP), soft, max(lo, 0), speech_db)
        else:
            back = int(INNER_START_WALK / HOP)
            while back > 0 and s0 > lo and audible[s0 - 1]:
                s0 -= 1; back -= 1
        if k == last_k:                              # outer end: let the last sound finish completely
            e0 = _walk(db, e0, +1, int(END_WALK / HOP), soft, hi, speech_db, voiced)
        else:
            fwd = int(INNER_END_WALK / HOP)
            while fwd > 0 and e0 < hi and e0 < n and audible[e0]:
                e0 += 1; fwd -= 1
        spans[k] = (s0, e0)
    a, b = spans[0][0], spans[-1][1]
    an.speech_start, an.speech_end = a * HOP, min(b * HOP, an.duration)

    # ── B: sound that no word explains ───────────────────────────────────────
    claimed = np.zeros(n, dtype=bool)
    guard = int(EVENT_GUARD / HOP)
    for s0, e0 in spans:
        claimed[max(s0 - guard, 0):min(e0 + guard, n)] = True
    left_over = audible & ~claimed
    inside, outside = [], []
    for s0, e0 in _runs(left_over):
        if (e0 - s0) * HOP < MIN_EVENT:
            continue
        rel = float(db[s0:e0].max()) - speech_db
        if e0 <= a:
            outside.append((s0 * HOP, e0 * HOP, "start", rel))
        elif s0 >= b:
            outside.append((s0 * HOP, e0 * HOP, "end", rel))
        else:
            inside.append((s0 * HOP, e0 * HOP, "inside", rel))
    if recording:
        if inside:
            an.status = "review"
            an.reasons.append(f"a sound between words at {inside[0][0]:.1f} s (breath, laugh or noise?) — check it")
        an.mute = []
    else:
        for ev in inside:
            dur = ev[1] - ev[0]
            loud = ev[3] > -LOUD_EVENT_DB
            if dur > MUTE_MAX or loud:
                an.status = "reject"
                an.reasons.append(f"{'loud ' if loud else ''}sound between words at {ev[0]:.1f} s ({dur:.2f} s)")
        an.mute = [(ev[0], ev[1]) for ev in inside]
    an.events = inside + outside

    # ── A: cut to the words, punctuation-sized margins that never include sound ──
    a = max(a - int(EXTRA_START / HOP), 0)          # a safety cushion on top of the measured edges,
    b = min(b + int(EXTRA_END / HOP), n)            # so a word's last few ms are never the thing cut
    lead_f, tail_f = int(LEAD / HOP), int(_tail_for(script) / HOP)
    cut0 = a
    for _ in range(lead_f):
        if cut0 <= 0 or audible[cut0 - 1]:
            break
        cut0 -= 1
    cut1 = b
    for _ in range(tail_f):
        if cut1 >= n or audible[cut1]:
            break
        cut1 += 1
    an.cut = (cut0 * HOP, min(cut1 * HOP, an.duration))
    an.aligned = True
    return an


def fades_for(an: Analysis, cut: tuple[float, float]) -> tuple[float, float]:
    """(fade-in, fade-out) seconds for a clip cut at `cut`: smooth, but only inside the cushion
    beyond the measured word edges, so the words themselves are never faded."""
    fin = min(FADE_IN, max(an.speech_start - cut[0], 0.005))
    fout = min(FADE_OUT, max(cut[1] - an.speech_end, 0.005))
    return fin, fout


def qualify_clip(path: Path, script: str, out_path: Path | None = None, verify: bool = True,
                 recording: bool = False) -> Result:
    """Qualify one inflated clip against its script. When the verdict is "ok" the trimmed (and
    gap-muted) audio is written to `out_path` (default: replaces `path`)."""
    res = Result()
    out_path = out_path or path
    an = analyze_clip(path, script, recording)
    res.score, res.words = an.score, an.words
    if not an.aligned or an.status in ("reject", "review"):
        res.status, res.reasons = an.status, an.reasons
        return res
    x, sr = sf.read(str(path), dtype="float32", always_2d=False)
    y = K.process_audio(x.mean(axis=1) if x.ndim > 1 else x, sr, 0.0, an.cut[0], an.cut[1],
                        tuple(an.mute), fade=fades_for(an, an.cut))
    res.trimmed_start, res.trimmed_end = an.cut[0], an.duration - an.cut[1]
    res.muted = len(an.mute)

    # ── Whisper must hear the script in what's left ──────────────────────────
    if verify:
        from faily.modules.asr import transcribe
        heard = transcribe(y, sr)
        ref, hyp = K.normalize_words(script), K.normalize_words(heard)
        res.wer = K.word_error_rate(ref, hyp)
        if res.wer >= WER_REJECT:
            res.status, res.reasons = "reject", [f"says “{heard.strip()[:60]}” instead of its script"]
            return res
        if res.wer > WER_OK:
            res.status, res.reasons = "review", [f"Whisper hears a different wording ({res.wer:.0%} of words)"]
            return res

    sf.write(str(out_path), y, sr)
    return res


def auto_qualify_stage(
    stage: str, progress: Callable[[int, int, str], None] | None = None, cancel: list[bool] | None = None,
) -> dict[str, Result]:
    """Auto-qualify freshly split RECORDING slices held in a staging character. Whisper writes each
    slice's transcript, the transcript is aligned to the audio, and a slice that aligns cleanly is cut
    to its words (rewritten in place). Anything unsure — alignment doubt, a sound between words,
    nothing heard — is left as it is and marked "review". Returns {clip key: Result}."""
    from faily.modules.asr import transcribe
    out: dict[str, Result] = {}
    try:
        clips = K.own_clips(stage)
        for i, c in enumerate(clips):
            if cancel and cancel[0]:
                break
            if progress:
                progress(i, len(clips), c.audio.stem)
            try:
                x, sr = sf.read(str(c.audio), dtype="float32", always_2d=False)
                text = transcribe(x.mean(axis=1) if x.ndim > 1 else x, sr).strip()
                if not text:
                    out[c.key] = Result(status="review", reasons=["no speech heard"])
                    continue
                K.set_transcript(stage, c, text)
                out[c.key] = qualify_clip(c.audio, text, verify=False, recording=True)
            except Exception as exc:
                out[c.key] = Result(status="review", reasons=[f"couldn't check: {exc}"])
    finally:
        unload()
        try:
            from faily.modules.asr import unload as _unload_asr
            _unload_asr()
        except Exception:
            pass
    return out


# ── the manual qualify screen uses the same analysis ─────────────────────────

def refine_audits(
    audits: dict[str, dict], clips: list[tuple[str, Path, str]],
    progress: Callable[[int, int, str], None] | None = None, cancel: list[bool] | None = None,
) -> None:
    """Make the qualify screen's findings for inflated clips come from the same analysis as
    AUTO-QUALIFY. `audits` (from inflation_audit.audit_character) is updated in place: its trim,
    noise events and noises-between-words are replaced with this module's; its script
    comparison, duplicate and background findings stay. `clips` = [(key, path, script)]."""
    from faily.modules.inflation_audit import Event
    try:
        for i, (key, path, script) in enumerate(clips):
            if cancel and cancel[0]:
                break
            if progress:
                progress(i, len(clips), path.stem)
            try:
                an = analyze_clip(path, script)
            except Exception as exc:
                an = Analysis(status="review", reasons=[f"couldn't align: {exc}"])
            audit = audits.get(key)
            if audit is None:
                audit = audits[key] = {"events": [], "duration": round(an.duration, 2), "snr_db": 0.0, "mute": [],
                                       "trim": None, "words": None, "recommend_delete": False, "reason": "",
                                       "issues": [], "edge_issues": [], "ok": True, "skipped": 0}
            if not an.aligned:                            # no alignment: keep the older findings, say why
                audit["issues"] = list(audit.get("issues", [])) + [f"auto-qualify can't verify the edges: {an.reasons[0]}"]
                audit["ok"] = False
                continue
            audit["aligned"] = True
            audit["score"] = round(an.score, 2)
            audit["issues"] = [i_ for i_ in audit.get("issues", []) if "between words" not in i_]
            audit["events"] = [Event(s, e, "sound between words" if w == "inside" else "sound outside the words", rel, w)
                               for s, e, w, rel in an.events]
            audit["mute"] = [(round(s, 2), round(e, 2)) for s, e in an.mute]
            audit["skipped"] = 0
            if an.mute:
                audit["issues"].append(f"{len(an.mute)} sound{'s' if len(an.mute) != 1 else ''} between words")
            if an.status == "reject":
                audit["issues"] += [r for r in an.reasons if r not in audit["issues"]]
                audit["recommend_delete"] = True
                audit["reason"] = audit.get("reason") or "there is a sound between its words that can't be removed cleanly"
            t0, t1 = an.cut
            lead, tail = t0, an.duration - t1
            before = an.speech_start                       # seconds before the first word
            after = an.duration - an.speech_end            # seconds after the last word
            edge = []
            if any(ev[2] == "start" for ev in an.events):
                edge.append("a stray noise before the first word")
            elif before > 0.25:
                edge.append(f"{before:.1f} s before the first word")
            if any(ev[2] == "end" for ev in an.events):
                edge.append("a stray noise after the last word")
            elif after > 0.25:
                edge.append(f"{after:.1f} s after the last word")
            audit["edge_issues"] = edge
            audit["trim"] = ({"trim_start": round(t0, 2), "trim_end": round(t1, 2), "duration": round(an.duration, 2),
                              "speech_start": round(an.speech_start, 2), "speech_end": round(an.speech_end, 2),
                              "fade": fades_for(an, an.cut)}
                             if lead > 0.005 or tail > 0.005 else None)
            audit["ok"] = not (audit["issues"] or edge)
    finally:
        unload()


def K_resample(mono: np.ndarray, sr: int) -> np.ndarray:
    from faily.modules.asr import _resample_16k
    return _resample_16k(mono, sr)


def qualify_batch(
    items: list[tuple[Path, str]], progress: Callable[[int, int], None] | None = None,
    cancel: list[bool] | None = None, verify: bool = True,
) -> list[Result]:
    """Qualify many (path, script) clips. Blocking; clips are rewritten in place when kept."""
    results: list[Result] = []
    try:
        for i, (path, script) in enumerate(items):
            if cancel and cancel[0]:
                break
            if progress:
                progress(i, len(items))
            try:
                results.append(qualify_clip(path, script, verify=verify))
            except Exception as exc:                # one bad file must not sink the batch
                r = Result(status="review", reasons=[f"couldn't qualify: {exc}"])
                results.append(r)
    finally:
        unload()
        try:
            from faily.modules.asr import unload as _unload_asr
            _unload_asr()
        except Exception:
            pass
    return results
