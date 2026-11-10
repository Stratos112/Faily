"""The reference-clip data model behind the REFERENCES hub, the ABOUT profile and the
training "activation" panel. No UI here.

A character's own clips are its primary reference (config "ref_audio") plus its "ref_clips".
Each clip has:
  origin   "recording" (uploaded / imported audio) or "inflated" (generated zero-shot)
  active   used when TRAINING a voice model (default True; the primary is always active)
  clone    usable as a zero-shot cloning reference (False = the older "excluded" flag)
  source   uploaded (added as-is), imported (from Split & qualify) or inflated; inflated clips also carry
           category / backend / batch (which script category, engine and wizard run). Older inflated clips
           lack backend/batch and show as "unknown".
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


SOURCE_LABELS = {"": "uploaded", "import": "imported", "inflated": "inflated"}


def source_label(source: str) -> str:
    """How a clip's `source` tag reads in the UI."""
    return SOURCE_LABELS.get(source, source or "uploaded")


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
    "source": "Source", "category": "Category", "backend": "Backend", "batch": "Batch",
    "length": "Length", "state": "Training state",
}
MEASURES = {"clips": "Clips", "minutes": "Minutes", "avg": "Avg length (s)"}


def _dim_value(r: dict, dim: str) -> str:
    if dim == "source":
        return source_label(r["source"] if not r["primary"] else "")
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


# ── the group tree ──────────────────────────────────────────────────────────

def _node(node_id: str, label: str, path: list[str], flt: dict, rs: list[dict]) -> dict:
    act = [r for r in rs if r["active"]]
    return {"id": node_id, "label": label, "path": path, "level": len(path), "filter": flt, "parent": "/".join(path[:-1]),
            "keys": [r["key"] for r in rs], "clips": len(rs), "seconds": sum(r["seconds"] for r in rs),
            "active_clips": len(act), "active_seconds": sum(r["seconds"] for r in act), "children": []}


def group_tree(rows: list[dict]) -> dict:
    """Every clip filed in a tree of groups:
         all data
         ├─ recordings ── uploaded, imported
         └─ inflated ──── <backend> ── <script category>
    `filter` on each node is the table filter that shows exactly that group's clips."""
    root = _node("", "all data", [], {}, rows)
    rec = [r for r in rows if r["origin"] == "recording"]
    inf = [r for r in rows if r["origin"] == "inflated"]
    if rec:
        n = _node("recordings", "recordings", ["recordings"], {"origin": "recording"}, rec)
        by: dict[str, list[dict]] = {}
        for r in rec:
            by.setdefault(source_label("" if r["primary"] else r["source"]), []).append(r)
        for lab in sorted(by, key=lambda x: (x != "uploaded", x)):
            n["children"].append(_node(f"recordings/{lab}", lab, ["recordings", lab], {"origin": "recording", "source": lab}, by[lab]))
        root["children"].append(n)
    if inf:
        n = _node("inflated", "inflated", ["inflated"], {"origin": "inflated"}, inf)
        by_backend: dict[str, list[dict]] = {}
        for r in inf:
            by_backend.setdefault(r["backend"] or UNKNOWN, []).append(r)
        for be in sorted(by_backend, key=lambda x: (x == UNKNOWN, x)):
            bn = _node(f"inflated/{be}", be, ["inflated", be], {"origin": "inflated", "backend": be}, by_backend[be])
            by_cat: dict[str, list[dict]] = {}
            for r in by_backend[be]:
                by_cat.setdefault(r["category"] or "uncategorised", []).append(r)
            for cat in sorted(by_cat):
                raw = by_cat[cat][0]["category"]
                bn["children"].append(_node(f"inflated/{be}/{cat}", cat, ["inflated", be, cat],
                                            {"origin": "inflated", "backend": be, "category": raw}, by_cat[cat]))
            n["children"].append(bn)
        root["children"].append(n)
    return root


def walk(node: dict):
    """The node and everything under it, parents first."""
    yield node
    for c in node["children"]:
        yield from walk(c)


def find_node(tree: dict, node_id: str) -> dict | None:
    return next((n for n in walk(tree) if n["id"] == node_id), None)


def leaves(node: dict) -> list[dict]:
    return [n for n in walk(node) if not n["children"] and n["clips"]]


# ── mixes (PARKED: tested, but not wired to any UI right now — the sliders came off the TRAINING tab on purpose) ────────────────────────────────────────────────────────────────────
# A mix says how the training audio is shared out. Each node has a WEIGHT among its siblings (they sum
# to 1); its SHARE of the whole is the product of the weights down to it. Moving a parent's share keeps
# its children's weights, so they shrink or grow with it; its siblings take the other side. Only a
# draft until apply_mix() switches clips on/off to match.

def mix_weights(tree: dict) -> dict[str, float]:
    """Weights that reproduce what is switched on right now. A branch with nothing on gets weights from
    its available audio instead, so switching it back on has sensible proportions."""
    w: dict[str, float] = {}

    def go(node):
        kids = node["children"]
        if not kids:
            return
        for field in ("active_seconds", "seconds"):
            total = sum(c[field] for c in kids)
            if total > 0:
                for c in kids:
                    w[c["id"]] = c[field] / total
                break
        else:
            for c in kids:
                w[c["id"]] = 1.0 / len(kids)
        for c in kids:
            go(c)
    go(tree)
    return w


def node_shares(tree: dict, weights: dict[str, float]) -> dict[str, float]:
    """Each node's share (0..1) of the whole mix."""
    share = {tree["id"]: 1.0}

    def go(node):
        for c in node["children"]:
            share[c["id"]] = share[node["id"]] * weights.get(c["id"], 0.0)
            go(c)
    go(tree)
    return share


def set_share(tree: dict, weights: dict[str, float], node_id: str, share: float) -> dict[str, float]:
    """New weights after setting `node_id`'s overall share. Its siblings rescale to fill the rest of the
    parent's share; its own children keep their proportions (so they scale with it). A node whose parent
    has no share can't be set (nothing to take it from)."""
    parent = next((n for n in walk(tree) if any(c["id"] == node_id for c in n["children"])), None)
    if parent is None:
        return dict(weights)
    parent_share = node_shares(tree, weights)[parent["id"]]
    if parent_share <= 1e-9:
        return dict(weights)
    sibs = [c for c in parent["children"] if c["id"] != node_id]
    rel = min(max(share / parent_share, 0.0), 1.0) if sibs else 1.0
    out = dict(weights)
    out[node_id] = rel
    rest = sum(weights.get(c["id"], 0.0) for c in sibs)
    for c in sibs:
        out[c["id"]] = weights.get(c["id"], 0.0) * (1 - rel) / rest if rest > 1e-9 else (1 - rel) / len(sibs)
    return out


def mix_plan(tree: dict, weights: dict[str, float], cap_seconds: float | None = None) -> dict:
    """How much audio each leaf group should contribute: as much total as the scarcest group allows
    (so the proportions hold), optionally capped. {"total", "targets": {leaf_id: seconds}}."""
    share = node_shares(tree, weights)
    live = [(n, share[n["id"]]) for n in leaves(tree) if share[n["id"]] > 1e-9]
    if not live:
        return {"total": 0.0, "targets": {n["id"]: 0.0 for n in leaves(tree)}}
    total = min(n["seconds"] / sh for n, sh in live)
    if cap_seconds:
        total = min(total, cap_seconds)
    return {"total": total, "targets": {n["id"]: (share[n["id"]] * total if share[n["id"]] > 1e-9 else 0.0) for n in leaves(tree)}}


def apply_mix(name: str, weights: dict[str, float], cap_seconds: float | None = None) -> dict:
    """Switch clips on/off so the active audio follows the mix. Within a group, clips already on are
    kept first (stable), the primary always stays on. Returns {"active_clips", "active_seconds", "total"}."""
    rows = clip_rows(name)
    tree = group_tree(rows)
    plan = mix_plan(tree, weights, cap_seconds)
    by_key = {r["key"]: r for r in rows}
    on: list[str] = []
    for leaf in leaves(tree):
        target = plan["targets"][leaf["id"]]
        order = sorted((by_key[k] for k in leaf["keys"]), key=lambda r: (not r["primary"], not r["active"]))
        got = 0.0
        for r in order:
            if r["primary"]:
                on.append(r["key"])
                got += r["seconds"]
            elif target > 0 and (got == 0 or got + r["seconds"] / 2 <= target):
                on.append(r["key"])              # nearest whole clip to the target, but never drop a group that has a share
                got += r["seconds"]
    keep = set(on)
    set_active(name, [k for k in by_key if k not in keep], False)
    set_active(name, on, True)
    after = [r for r in clip_rows(name) if r["active"]]
    return {"active_clips": len(after), "active_seconds": sum(r["seconds"] for r in after), "total": plan["total"]}


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
