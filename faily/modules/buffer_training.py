"""Buffer training — bulk-generate zero-shot clips of a character's own voice
across a phonetically/prosodically diverse script bank, to pad out their
reference pool before Piper training. Pure logic; no UI imports.
"""
import asyncio
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from faily.core.characters import build_ref_audio, clip_quality_issues
from faily.modules.vc import generate

_SCRIPT_BANK_PATH = Path("faily/data/buffer_scripts.txt")

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
class BufferCandidate:
    category: str
    text: str
    path: Path
    issues: list[str] = field(default_factory=list)
    accepted: bool = True


def load_script_bank(path: Path = _SCRIPT_BANK_PATH) -> dict[str, list[str]]:
    """Parse the buffer-scripts text file into {category_key: [line, ...]}.

    "## CATEGORY_KEY" header lines start a category; blank lines are ignored;
    lines starting with "#" outside a header are comments. Raises
    FileNotFoundError if the bank is missing — the caller should show_error()
    and not open the wizard, since this is a packaging bug, not a user error.
    """
    if not path.exists():
        raise FileNotFoundError(f"Script bank not found: {path}")
    bank: dict[str, list[str]] = {}
    current: str | None = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
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


def sample_lines(
    bank: dict[str, list[str]],
    weights: dict[str, float],
    count: int,
    rng: random.Random | None = None,
) -> list[ScriptLine]:
    """Weighted draw of up to `count` lines across categories.

    Categories with weight <= 0 or an empty pool are excluded entirely (an
    explicit "none of this", not redistributed to others). Allocation across
    the remaining categories uses largest-remainder apportionment so the
    total is exact. Within a category, if the allocation exceeds its pool
    size, the pool is shuffled and repeated in full passes (no identical
    line repeats until every line in the pool has been used once) rather
    than sampled with replacement, which could repeat the same line
    back-to-back by chance. The combined result is globally shuffled before
    returning, so an early stop during generation still yields roughly
    proportional category coverage instead of exhausting categories in a
    fixed order.
    """
    rng = rng or random.Random()
    usable_weights = {k: w for k, w in weights.items() if w > 0 and bank.get(k)}
    allocation = _allocate_counts(usable_weights, count)

    lines: list[ScriptLine] = []
    for category, k in allocation.items():
        pool = bank[category]
        if k <= len(pool):
            drawn = rng.sample(pool, k)
        else:
            drawn = []
            while len(drawn) < k:
                pass_copy = pool[:]
                rng.shuffle(pass_copy)
                drawn.extend(pass_copy)
            drawn = drawn[:k]
        lines.extend(ScriptLine(category=category, text=text) for text in drawn)

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
) -> list[BufferCandidate]:
    """Sequentially generate one zero-shot clip per script line, using the
    character's own existing reference audio as the zero-shot voice
    reference. Checks cancel_flag before starting each clip (an in-flight
    generation can't be interrupted, only the next one can be skipped).
    A single clip's failure is caught and reported via on_error without
    aborting the rest of the batch.
    """
    staging_dir.mkdir(parents=True, exist_ok=True)
    candidates: list[BufferCandidate] = []
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
                candidates.append(BufferCandidate(
                    category=line.category, text=line.text, path=path, issues=issues,
                ))
            except Exception as exc:
                on_error(i, line, exc)
            on_progress(i + 1, total, line)
    return candidates
