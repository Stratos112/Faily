"""Data inflation — bulk-generate zero-shot clips of a character's own voice
across a phonetically/prosodically diverse script bank, to inflate their
reference pool before Piper training. Pure logic; no UI imports.
"""
import asyncio
import math
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from faily.core.characters import build_ref_audio, clip_quality_issues, get_character
from faily.modules.vc import generate

_SCRIPT_BANK_PATH = Path("faily/data/inflation_scripts.txt")

# key -> {"label": str, "desc": str} — same shape as vc.BACKENDS, so Step 1's
# slider builder can iterate this the same way model_picker iterates BACKENDS.
CATEGORIES: dict[str, dict] = {
    "plosives": {
        "label": "Plosives & Attack",
        "desc": "Stops (p/b/t/d/k/g) — sharp articulation onset, transient burst energy.",
    },
    "sibilance": {
        "label": "Sibilance & Fricatives",
        "desc": "s/z/sh/zh/f/v/th/h — high-frequency continuant energy, perceived sharpness/clarity.",
    },
    "nasals": {
        "label": "Nasals & Liquids",
        "desc": "m/n/ng/l/r/w/y — resonance and smooth articulatory transitions.",
    },
    "vowels": {
        "label": "Vowel Space Breadth",
        "desc": "Front/back, open/close vowels + diphthongs — the biggest lever on timbre/formant fidelity.",
    },
    "clusters": {
        "label": "Consonant Clusters & Complex Words",
        "desc": "Cluster-heavy and multisyllabic words — articulation and duration stress-test.",
    },
    "pacing": {
        "label": "Sentence Length & Pacing Variety",
        "desc": "Short punchy vs. long compound sentences, rapid enumeration vs. slow deliberate clauses.",
    },
    "intonation": {
        "label": "Intonation Variety",
        "desc": "Statements, questions, exclamations, commands — pitch-contour range.",
    },
    "numbers": {
        "label": "Numbers & Normalization",
        "desc": "Digits, dates, currency, abbreviations, units — text-normalization coverage.",
    },
    "neutral": {
        "label": "Neutral / Everyday Register",
        "desc": "Plain, unmarked conversational sentences — the diversity control group.",
    },
}


@dataclass
class ScriptLine:
    category: str
    text: str


@dataclass
class InflatedCandidate:
    category: str
    text: str
    path: Path
    issues: list[str] = field(default_factory=list)
    accepted: bool = True
    verdict: str = ""            # after auto-qualify: "ok" | "review" | "reject"
    note: str = ""               # what auto-qualify did / found


def load_script_bank(path: Path = _SCRIPT_BANK_PATH) -> dict[str, list[str]]:
    """Parse the inflation-scripts text file (plus any inflation_scripts_*.txt beside
    it) into {category_key: [line, ...]}.

    "## CATEGORY_KEY" header lines start a category; blank lines are ignored;
    lines starting with "#" outside a header are comments. Raises
    FileNotFoundError if the bank is missing — the caller should show_error()
    and not open the wizard, since this is a packaging bug, not a user error.
    """
    if not path.exists():
        raise FileNotFoundError(f"Script bank not found: {path}")
    # Extra banks sit beside the main one as inflation_scripts_<name>.txt (same
    # format) and are merged in; sample_lines() drops cross-file duplicates.
    paths = [path, *sorted(path.parent.glob(f"{path.stem}_*{path.suffix}"))]
    bank: dict[str, list[str]] = {}
    for file in paths:
        current: str | None = None
        for raw_line in file.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("##"):
                current = line[2:].strip()
                bank.setdefault(current, [])
                continue
            if line.startswith("#"):
                continue
            if current is None:
                continue
            bank[current].append(line)
    return bank


# ── how a line starts ─────────────────────────────────────────────────────────
# The first sound matters: a voice trained on lines that mostly open with "The" or a hard
# stop learns to open that way, and has trouble starting a sentence softly. sample_lines()
# therefore draws the opening sounds in these proportions (as far as the bank allows)
# instead of however they happen to occur in the bank.
START_WEIGHTS: dict[str, float] = {
    "vowel": 0.25,        # a, e, i, o, u...
    "sonorant": 0.30,     # m n l r w y h — soft consonants
    "fricative": 0.20,    # s z f v sh
    "th": 0.10,           # "the", "this", "that", "three"
    "stop": 0.15,         # p b t d k g ch j — hard onsets
}
_TH_WORDS = {"the", "this", "that", "these", "those", "there", "then", "they", "their", "them", "than", "though", "thus",
             "thin", "think", "thirty", "three", "through", "thick", "thousand", "thank", "thanks", "third", "thing", "things",
             "thought", "throw", "threw", "thumb", "thunder", "thursday", "thirteen", "theory", "therefore", "these"}
_SILENT_START = ("kn", "wr", "gn", "ps", "wh")
_VOWEL_SOUND_H = {"hour", "honest", "heir", "honor", "honour"}


def start_class(text: str) -> str:
    """The class of a line's first sound, from its spelling (a close approximation)."""
    m = re.search(r"[A-Za-z']+", text)
    w = (m.group(0) if m else "").lower().strip("'")
    if not w:
        return "sonorant"                         # starts with a digit / symbol: spoken words, mostly soft
    if w in _TH_WORDS or w.startswith("th"):
        return "th"
    if w in _VOWEL_SOUND_H:
        return "vowel"
    if w.startswith(("one", "once")):
        return "sonorant"                         # "w"
    if w.startswith(("use", "usu", "uni", "eu", "ew")):
        return "sonorant"                         # "y"
    if w.startswith(_SILENT_START):
        return "sonorant"
    c = w[0]
    if c in "aeiou":
        return "vowel"
    if c in "mnlrwyh":
        return "sonorant"
    if c in "fvszx" or w.startswith(("sh", "ph", "ce", "ci", "cy")):
        return "fricative"
    return "stop"                                 # p b t d k g c ch j q


def _balanced_order(texts: list[str], rng: random.Random) -> list[str]:
    """`texts` in the order they should be drawn: each draw picks a start class by START_WEIGHTS
    (among classes with lines left), then a first word at random within it, then one of that
    word's lines — so no single opening ("The...") can dominate."""
    groups: dict[str, dict[str, list[str]]] = {}
    for t in texts:
        m = re.search(r"[A-Za-z']+", t)
        word = (m.group(0) if m else "").lower()
        groups.setdefault(start_class(t), {}).setdefault(word, []).append(t)
    for words in groups.values():
        for lst in words.values():
            rng.shuffle(lst)
    out: list[str] = []
    while groups:
        classes = list(groups)
        cls = rng.choices(classes, weights=[START_WEIGHTS.get(c, 0.1) for c in classes])[0]
        words = groups[cls]
        word = rng.choice(list(words))
        out.append(words[word].pop())
        if not words[word]:
            del words[word]
        if not words:
            del groups[cls]
    return out


def _allocate_counts(weights: dict[str, float], count: int) -> dict[str, int]:
    """Largest-remainder (Hamilton) apportionment of `count` across categories
    proportional to `weights`. Categories with weight <= 0 are excluded, not
    redistributed. Guarantees the returned counts sum to exactly `count`
    (or 0 if every weight is <= 0)."""
    active = {k: w for k, w in weights.items() if w > 0}
    total_weight = sum(active.values())
    if total_weight <= 0 or count <= 0:
        return {}
    raw = {k: (w / total_weight) * count for k, w in active.items()}
    floors = {k: math.floor(v) for k, v in raw.items()}
    remainder = count - sum(floors.values())
    order = sorted(active.keys(), key=lambda k: raw[k] - floors[k], reverse=True)
    for k in order[:remainder]:
        floors[k] += 1
    return floors


def _norm(text: str) -> str:
    """Key for "is this the same line": whitespace- and case-insensitive."""
    return " ".join(text.split()).casefold()


def used_inflated_lines(char_name: str) -> set[str]:
    """Normalized text of every script line already committed as an inflated
    clip for this character or any ancestor (a sub-character's reference
    chain includes its parents' clips). Excluded clips count: the line was
    still used. Pass the result to sample_lines(exclude=...)."""
    used: set[str] = set()
    seen: set[str] = set()
    current: str | None = char_name
    while current and current not in seen:
        seen.add(current)
        char = get_character(current)
        if not char:
            break
        for rc in char.get("ref_clips", []):
            if rc.get("source") == "inflated" and rc.get("transcript"):
                used.add(_norm(rc["transcript"]))
        current = char.get("parent")
    return used


def sample_lines(
    bank: dict[str, list[str]],
    weights: dict[str, float],
    count: int,
    rng: random.Random | None = None,
    exclude: set[str] | None = None,
) -> list[ScriptLine]:
    """Weighted draw of up to `count` DISTINCT lines across categories.

    No line is ever returned twice, and none in `exclude` (normalized texts —
    see used_inflated_lines) is returned at all. Duplicates inside the bank
    itself (same text in two places) count as one line.

    Allocation across categories with weight > 0 uses largest-remainder
    apportionment so the total is exact. When a category has fewer unused
    lines than its share, the shortfall is re-apportioned across the other
    weighted categories that still have unused lines, repeating until the
    count is met. If every weighted category runs out, fewer than `count`
    lines come back — the caller should say so. Categories with weight <= 0
    stay excluded: an explicit "none of this" is never redistributed into.
    The combined result is globally shuffled, so an early stop during
    generation still yields roughly proportional category coverage.
    """
    rng = rng or random.Random()
    taken = set(exclude or ())
    pools: dict[str, list[str]] = {}
    for category, texts in bank.items():
        fresh = []
        for text in texts:
            key = _norm(text)
            if key not in taken:
                taken.add(key)
                fresh.append(text)
        # popped from the end, so reverse the preferred order
        pools[category] = _balanced_order(fresh, rng)[::-1]

    usable = {k: w for k, w in weights.items() if w > 0 and pools.get(k)}
    want = _allocate_counts(usable, count)
    lines: list[ScriptLine] = []
    while want:
        shortfall = 0
        for category, k in want.items():
            pool = pools[category]
            take = min(k, len(pool))
            lines.extend(ScriptLine(category=category, text=pool.pop()) for _ in range(take))
            shortfall += k - take
        candidates = {k: w for k, w in usable.items() if pools[k]}
        want = _allocate_counts(candidates, shortfall) if shortfall and candidates else {}

    rng.shuffle(lines)
    return lines


async def generate_batch(
    char_name: str,
    lines: list[ScriptLine],
    backend: str,
    param1: float,
    param2: float,
    staging_dir: Path,
    cancel_flag: list[bool],
    on_progress: Callable[[int, int, ScriptLine], None],
    on_error: Callable[[int, ScriptLine, Exception], None],
) -> list[InflatedCandidate]:
    """Sequentially generate one zero-shot clip per script line, using the
    character's own existing reference audio as the zero-shot voice
    reference. Checks cancel_flag before starting each clip (an in-flight
    generation can't be interrupted, only the next one can be skipped).
    A single clip's failure is caught and reported via on_error without
    aborting the rest of the batch.
    """
    staging_dir.mkdir(parents=True, exist_ok=True)
    candidates: list[InflatedCandidate] = []
    require_transcript = backend == "f5_tts"
    max_duration = 15.0 if backend == "f5_tts" else None

    with build_ref_audio(char_name, require_transcript=require_transcript, max_duration=max_duration) as (ref_path, ref_text):
        total = len(lines)
        for i, line in enumerate(lines):
            if cancel_flag[0]:
                break
            try:
                path = await asyncio.to_thread(
                    generate, line.text, ref_path, None, staging_dir,
                    backend, param1, param2, ref_text, char_name,
                )
                issues = clip_quality_issues(path)
                candidates.append(InflatedCandidate(
                    category=line.category, text=line.text, path=path, issues=issues,
                ))
            except Exception as exc:
                on_error(i, line, exc)
            on_progress(i + 1, total, line)
    return candidates
