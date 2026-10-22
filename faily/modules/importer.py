"""Import a recording as reference clips: slice it, review the slices in a hidden
staging character, then add the survivors to a new or existing character.

The staging character (outputs/characters/_stage_<id>/, hidden from every
character list) lets the review reuse all of modules/qualify.py unchanged; nothing
touches a real character until commit_stage().
"""
from __future__ import annotations

import json
import shutil
import time
import uuid
from datetime import datetime
from pathlib import Path

from faily.core import characters as C
from faily.modules.slicer import Slice

STAGE_PREFIX = "_stage_"


def purge_old_stages(max_age_hours: float = 24.0) -> None:
    """Remove staging directories abandoned by a closed browser tab."""
    if not C.CHARACTERS_DIR.is_dir():
        return
    cutoff = time.time() - max_age_hours * 3600
    for d in C.CHARACTERS_DIR.glob(f"{STAGE_PREFIX}*"):
        try:
            if d.is_dir() and d.stat().st_mtime < cutoff:
                shutil.rmtree(d, ignore_errors=True)
        except OSError:
            pass


def stage_slices(slices: list[tuple[Path, Slice]], source: Path) -> str:
    """Create a staging character holding `slices` (copied in) as ref clips with empty
    transcripts. Returns its name."""
    purge_old_stages()
    name = f"{STAGE_PREFIX}{uuid.uuid4().hex[:8]}"
    d = C.CHARACTERS_DIR / name
    (d / "refs").mkdir(parents=True)
    clips = []
    for i, (wav, sl) in enumerate(slices, 1):
        dest = d / "refs" / f"ref_{i:03d}.wav"
        shutil.copy2(wav, dest)
        clips.append({
            "file": f"refs/{dest.name}", "transcript": "", "source": "import",
            "from": source.name, "start": round(sl.start, 2), "end": round(sl.end, 2),
        })
    (d / "config.json").write_text(json.dumps({
        "name": name, "created": datetime.now().isoformat(), "ref_clips": clips,
    }, indent=2))
    return name


def stage_inflated(candidates: list, name_hint: str = "") -> str:
    """Stage freshly inflated clips (InflatedCandidates) so they can go through the
    same qualification as an import before joining the pool. Their script is the transcript;
    source/category are kept so they are still inflated data after commit_stage()."""
    purge_old_stages()
    name = f"{STAGE_PREFIX}{uuid.uuid4().hex[:8]}"
    d = C.CHARACTERS_DIR / name
    (d / "refs").mkdir(parents=True)
    clips = []
    for i, cand in enumerate(candidates, 1):
        dest = d / "refs" / f"ref_{i:03d}{cand.path.suffix}"
        shutil.copy2(cand.path, dest)
        clips.append({
            "file": f"refs/{dest.name}", "transcript": " ".join(cand.text.split()),
            "source": "inflated", "category": cand.category,
        })
    (d / "config.json").write_text(json.dumps({
        "name": name, "created": datetime.now().isoformat(), "ref_clips": clips,
    }, indent=2))
    return name


def discard_stage(stage: str) -> None:
    if stage.startswith(STAGE_PREFIX):
        shutil.rmtree(C.CHARACTERS_DIR / stage, ignore_errors=True)


def commit_stage(stage: str, target: str, is_new: bool) -> dict:
    """Move the staged clips into character `target`.
    New character: its primary reference is the first clip and the rest are ref clips.
    Existing: every clip is added as a ref clip. Returns counts and deletes the stage."""
    from faily.modules.qualify import own_clips

    clips = own_clips(stage)
    # inflated clips staged by stage_inflated() stay inflated (source/category); imports are "import"
    tags = {e["audio"]: (e.get("source") or "import", e.get("category", ""))
            for e in C.get_ref_chain(stage) if e["owner"] == stage}
    if not clips:
        discard_stage(stage)
        return {"added": 0, "no_transcript": 0}
    if is_new:
        if C.get_character(target) is not None:
            raise FileExistsError(f"A character called '{target}' already exists")
        first, rest = clips[0], clips[1:]
        C.save_character(target, first.audio, first.transcript)
    else:
        if C.get_character(target) is None:
            raise FileNotFoundError(f"Character '{target}' not found")
        rest = clips
    for c in rest:
        source, category = tags.get(c.audio, ("import", ""))
        C.add_ref_clip(target, c.audio, c.transcript, source=source, category=category)
    result = {"added": len(clips), "no_transcript": sum(1 for c in clips if not c.transcript.strip())}
    discard_stage(stage)
    return result
