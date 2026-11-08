"""The reference-clip data model behind the REFERENCES hub, the ABOUT profile and the
training "activation" panel. No UI here.

A character's own clips are its primary reference (config "ref_audio") plus its "ref_clips".
Each clip has:
  origin   "recording" (uploaded / imported audio) or "inflated" (generated zero-shot)
  active   used when TRAINING a voice model (default True; the primary is always active)
  clone    usable as a zero-shot cloning reference (False = the older "excluded" flag)
  category / backend / batch   provenance of inflated clips (which script category, which
           engine, which wizard run). Older clips lack backend/batch and show as "unknown".
Deleting moves clips to refs_trash/<id>/ with an index, so it can be undone.
"""
import json
import shutil
import time
from pathlib import Path

from faily.core import characters as C

PRIMARY = "__primary__"            # key of a character's ref_audio (it has no ref_clips entry)
LONG_FLOOR_SECONDS = 20.0          # a clip is "long" when over max(this, 3x the average)
LENGTH_BUCKETS = [(0, 2, "< 2 s"), (2, 4, "2–4 s"), (4, 6, "4–6 s"), (6, 8, "6–8 s"),
                  (8, 12, "8–12 s"), (12, 20, "12–20 s"), (20, 1e9, "20 s +")]
UNKNOWN = "unknown"

_DUR_CACHE: dict[tuple[str, int], float] = {}


def origin_of(source: str) -> str:
    return "inflated" if source == "inflated" else "recording"


def clip_seconds(path: Path) -> float:
    """Duration of an audio file, cached by (path, mtime) so listing thousands of clips is cheap."""
    import soundfile as sf
    try:
        key = (str(path), path.stat().st_mtime_ns)
    except OSError:
        return 0.0
    if key not in _DUR_CACHE:
        try:
            _DUR_CACHE[key] = float(sf.info(str(path)).duration)
        except Exception:                                    # noqa: BLE001  unreadable audio counts as 0 s
            _DUR_CACHE[key] = 0.0
    return _DUR_CACHE[key]


# ── rows ──────────────────────────────────────────────────────────────────────

def clip_rows(name: str) -> list[dict]:
    """The character's own clips, primary first. Missing audio files are skipped."""
    char = C.get_character(name)
    if not char:
        return []
    d = C.CHARACTERS_DIR / name
    rows: list[dict] = []
    if "ref_audio" in char and (d / char["ref_audio"]).exists():
        a = d / char["ref_audio"]
        rows.append({"key": PRIMARY, "file": char["ref_audio"], "audio": a, "name": a.stem, "primary": True,
                     "transcript": char.get("transcript", ""), "source": "", "origin": "recording",
                     "category": "", "backend": "", "batch": "", "added": "",
                     "active": True, "clone": True, "seconds": clip_seconds(a)})
    for rc in char.get("ref_clips", []):
        a = d / rc["file"]
        if not a.exists():
            continue
        src = rc.get("source", "")
        rows.append({"key": rc["file"], "file": rc["file"], "audio": a, "name": a.stem, "primary": False,
                     "transcript": rc.get("transcript", ""), "source": src, "origin": origin_of(src),
                     "category": rc.get("category", ""), "backend": rc.get("backend", ""),
                     "batch": rc.get("batch", ""), "added": rc.get("added", ""),
                     "active": rc.get("active", True), "clone": not rc.get("excluded", False),
                     "seconds": clip_seconds(a)})
    return rows


def inherited_rows(name: str) -> list[dict]:
    """Clips a variant gets from its parent chain (read-only here; manage them on the parent)."""
    own = {r["audio"] for r in clip_rows(name)}
    out = []
    for e in C.get_ref_chain(name, include_excluded=True):
        if e["audio"] in own or e["owner"] == name:
            continue
        out.append({**e, "name": e["audio"].stem, "origin": origin_of(e.get("source", "")),
                    "seconds": clip_seconds(e["audio"]), "clone": True})
    return out


def long_threshold(rows: list[dict]) -> float:
    """Seconds above which a clip counts as long for this character (3x its own average, at least 20 s)."""
    secs = [r["seconds"] for r in rows if r["seconds"] > 0]
    return max(LONG_FLOOR_SECONDS, 3.0 * sum(secs) / len(secs)) if secs else LONG_FLOOR_SECONDS


# ── statistics ───────────────────────────────────────────────────────────────

def _agg(rows: list[dict]) -> dict:
    n = len(rows)
    sec = sum(r["seconds"] for r in rows)
    return {"clips": n, "seconds": sec, "avg": sec / n if n else 0.0}


def summarize(rows: list[dict]) -> dict:
    """Totals and per-origin / per-category / per-backend figures, for all clips and for the active ones."""
    active = [r for r in rows if r["active"]]
    inflated = [r for r in rows if r["origin"] == "inflated"]

    def by(key, subset):
        out: dict[str, dict] = {}
        for r in subset:
            out.setdefault(r[key] or UNKNOWN, []).append(r)
        return {k: _agg(v) for k, v in sorted(out.items())}

    with_text = sum(1 for r in rows if r["transcript"].strip())
    return {
        "all": _agg(rows), "active": _agg(active),
        "recordings": _agg([r for r in rows if r["origin"] == "recording"]),
        "inflated": _agg(inflated),
        "active_recordings": _agg([r for r in active if r["origin"] == "recording"]),
        "active_inflated": _agg([r for r in active if r["origin"] == "inflated"]),
        "by_category": by("category", inflated), "by_backend": by("backend", inflated),
        "by_batch": by("batch", inflated),
        "transcribed": with_text, "transcript_share": with_text / len(rows) if rows else 0.0,
        "long_threshold": long_threshold(rows),
        "long": sum(1 for r in rows if r["seconds"] > long_threshold(rows)),
    }


DIMENSIONS = {          # key -> label, in the order the chart picker shows them
    "origin": "Source", "category": "Category", "backend": "Backend", "batch": "Batch",
    "length": "Length", "state": "Training state",
}
MEASURES = {"clips": "Clips", "minutes": "Minutes", "avg": "Avg length (s)"}


def _dim_value(r: dict, dim: str) -> str:
    if dim == "origin":
        return "inflated" if r["origin"] == "inflated" else "recordings"
    if dim == "length":
        s = r["seconds"]
        return next(label for lo, hi, label in LENGTH_BUCKETS if lo <= s < hi)
    if dim == "state":
        return "active" if r["active"] else "inactive"
    value = r[dim]
    if dim in ("category", "backend", "batch") and r["origin"] != "inflated":
        return "— (recordings)"
    return value or UNKNOWN


def breakdown(rows: list[dict], by: str, measure: str = "clips", stack: str | None = None) -> dict:
    """{"categories": [...], "series": [{"name", "data": [...]}]} for a bar chart: one bar per `by`
    value, optionally split into stacked segments by `stack`. measure = clips | minutes | avg."""
    if by not in DIMENSIONS or measure not in MEASURES:
        raise ValueError(f"unknown dimension/measure: {by!r}/{measure!r}")
    cells: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        cells.setdefault((_dim_value(r, by), _dim_value(r, stack) if stack else "all"), []).append(r)
    if by == "length":
        order = [label for *_, label in LENGTH_BUCKETS]
    else:
        order = sorted({c for c, _ in cells}, key=lambda c: (not c.startswith("—"), c))      # recordings' "—" first
    present = [c for c in order if any(k[0] == c for k in cells)]
    segments = sorted({s for _, s in cells})

    def value(rs: list[dict]) -> float:
        if measure == "clips":
            return float(len(rs))
        sec = sum(r["seconds"] for r in rs)
        if measure == "minutes":
            return round(sec / 60, 2)
        return round(sec / len(rs), 2) if rs else 0.0

    return {"categories": present,
            "series": [{"name": seg if stack else MEASURES[measure],
                        "data": [value(cells.get((c, seg), [])) for c in present]} for seg in segments]}


def groups(rows: list[dict]) -> list[dict]:
    """One entry per (origin, category, backend) — what the management cards show."""
    cells: dict[tuple[str, str, str], list[dict]] = {}
    for r in rows:
        cells.setdefault((r["origin"], r["category"], r["backend"] or (UNKNOWN if r["origin"] == "inflated" else "")), []).append(r)
    out = []
    for (origin, category, backend), rs in sorted(cells.items()):
        act = [r for r in rs if r["active"]]
        out.append({"origin": origin, "category": category, "backend": backend,
                    "keys": [r["key"] for r in rs], "clips": len(rs), "seconds": sum(r["seconds"] for r in rs),
                    "active_clips": len(act), "active_seconds": sum(r["seconds"] for r in act)})
    return out


# ── changing things ──────────────────────────────────────────────────────────

def _edit_config(name: str, fn) -> int:
    p = C._cfg(name)
    cfg = json.loads(p.read_text())
    n = fn(cfg)
    p.write_text(json.dumps(cfg, indent=2))
    return n


def set_active(name: str, keys: list[str], active: bool) -> int:
    """Switch clips on/off for TRAINING. The primary reference can't be switched off. Returns how many changed."""
    want = set(keys) - {PRIMARY}

    def go(cfg):
        n = 0
        for rc in cfg.get("ref_clips", []):
            if rc["file"] in want and rc.get("active", True) != active:
                if active:
                    rc.pop("active", None)                       # on is the default: keep configs tidy
                else:
                    rc["active"] = False
                n += 1
        return n
    return _edit_config(name, go)


def set_cloning(name: str, keys: list[str], enabled: bool) -> int:
    """Allow/forbid clips as zero-shot cloning references (the older 'excluded' flag, inverted)."""
    want = set(keys) - {PRIMARY}

    def go(cfg):
        n = 0
        for rc in cfg.get("ref_clips", []):
            if rc["file"] in want and (not rc.get("excluded", False)) != enabled:
                if enabled:
                    rc.pop("excluded", None)
                else:
                    rc["excluded"] = True
                n += 1
        return n
    return _edit_config(name, go)


def _trash_index(name: str) -> Path:
    return C.CHARACTERS_DIR / name / "refs_trash" / "index.json"


def _read_trash(name: str) -> list[dict]:
    try:
        return json.loads(_trash_index(name).read_text())
    except (OSError, ValueError):
        return []


def delete_clips(name: str, keys: list[str]) -> str | None:
    """Move clips to refs_trash/<id>/ (recoverable) and drop them from the character.
    The primary reference is never deleted here. Returns the trash id, or None if nothing moved."""
    want = set(keys) - {PRIMARY}
    d = C.CHARACTERS_DIR / name
    stamp = time.strftime("%Y%m%d-%H%M%S") + f"-{int(time.time() * 1000) % 1000:03d}"
    taken = {b["id"] for b in _read_trash(name)}
    tid, n = stamp, 1
    while tid in taken or (d / "refs_trash" / tid).exists():          # two deletes in one millisecond must not share a folder
        tid, n = f"{stamp}-{n}", n + 1
    moved: list[dict] = []

    def go(cfg):
        keep = []
        for rc in cfg.get("ref_clips", []):
            if rc["file"] in want:
                src = d / rc["file"]
                dest = d / "refs_trash" / tid / Path(rc["file"]).name
                dest.parent.mkdir(parents=True, exist_ok=True)
                if src.exists():
                    shutil.move(str(src), str(dest))
                moved.append({"entry": rc, "trashed": f"{tid}/{dest.name}"})
            else:
                keep.append(rc)
        cfg["ref_clips"] = keep
        return len(moved)

    if not _edit_config(name, go):
        return None
    index = _read_trash(name) + [{"id": tid, "at": time.time(), "items": moved}]
    _trash_index(name).write_text(json.dumps(index, indent=2))
    return tid


def last_deleted(name: str) -> dict | None:
    index = _read_trash(name)
    return index[-1] if index else None


def restore_deleted(name: str, trash_id: str) -> int:
    """Put a deletion back (audio and config entries). Returns how many clips came back."""
    d = C.CHARACTERS_DIR / name
    index = _read_trash(name)
    batch = next((b for b in index if b["id"] == trash_id), None)
    if not batch:
        return 0
    restored = []
    for item in batch["items"]:
        src = d / "refs_trash" / item["trashed"]
        target = d / item["entry"]["file"]
        if src.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(target))
            restored.append(item["entry"])

    def go(cfg):
        have = {rc["file"] for rc in cfg.get("ref_clips", [])}
        cfg.setdefault("ref_clips", []).extend(e for e in restored if e["file"] not in have)
        return len(restored)

    n = _edit_config(name, go)
    _trash_index(name).write_text(json.dumps([b for b in index if b["id"] != trash_id], indent=2))
    try:
        (d / "refs_trash" / trash_id).rmdir()
    except OSError:
        pass
    return n


def balance_active(name: str, inflated_share: float) -> dict:
    """Make roughly `inflated_share` (0..0.99) of the ACTIVE audio inflated. Every recording is switched
    on; inflated clips are then switched on round-robin across their categories until the share is
    reached (nearest whole clip), and the rest are switched off. For "everything on", use set_active.
    Returns {"active_clips", "active_seconds", "share"} describing the result."""
    share = min(max(inflated_share, 0.0), 0.99)
    rows = clip_rows(name)
    rec = [r for r in rows if r["origin"] == "recording"]
    inf = [r for r in rows if r["origin"] == "inflated"]
    rec_sec = sum(r["seconds"] for r in rec)
    want_sec = rec_sec * share / (1 - share) if rec_sec else 0.0
    if not rec:                                                 # nothing real to balance against
        want_sec = sum(r["seconds"] for r in inf) * share
    lanes: dict[str, list[dict]] = {}
    for r in inf:
        lanes.setdefault(r["category"], []).append(r)
    chosen, total = [], 0.0
    while total < want_sec and any(lanes.values()):
        for cat in sorted(lanes):
            if lanes[cat] and total < want_sec:
                r = lanes[cat].pop(0)
                chosen.append(r["key"])
                total += r["seconds"]
    set_active(name, [r["key"] for r in rec], True)
    set_active(name, [r["key"] for r in inf if r["key"] not in set(chosen)], False)
    set_active(name, chosen, True)
    after = [r for r in clip_rows(name) if r["active"]]
    sec = sum(r["seconds"] for r in after)
    inf_sec = sum(r["seconds"] for r in after if r["origin"] == "inflated")
    return {"active_clips": len(after), "active_seconds": sec, "share": inf_sec / sec if sec else 0.0}
