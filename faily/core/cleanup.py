"""Disk cleanup for a Faily outputs folder (the active one, or any other you point at).
Everything offered is regenerable or disposable; references, inflated clips, favorites,
the active model and character configs are never touched."""
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass
class Library:
    characters: Path | None
    foley: Path | None
    misc: Path | None


def open_library(root: Path | str) -> Library:
    """Accepts an outputs/ folder or a bare characters folder; refuses anything else."""
    root = Path(root).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"Not a folder: {root}")
    chars, foley, misc = root / "characters", root / "foley", root / "misc"
    if chars.is_dir() or foley.is_dir() or misc.is_dir():
        return Library(chars if chars.is_dir() else None,
                       foley if foley.is_dir() else None, misc if misc.is_dir() else None)
    if any((d / "config.json").exists() for d in root.iterdir() if d.is_dir()):
        return Library(root, None, None)
    raise ValueError("That doesn't look like a Faily outputs or characters folder")


def _characters(lib: Library):
    if lib.characters:
        for d in sorted(lib.characters.iterdir()):
            if d.is_dir() and (d / "config.json").exists():
                yield d


def _training_active(char_dir: Path) -> bool:
    try:
        from faily.modules.piper import read_training_status, _ACTIVE_STATES
        st = read_training_status(char_dir)
        return bool(st and st.get("state") in _ACTIVE_STATES)
    except Exception:
        return False


def _per_char(*names: str, skip_training: bool = False):
    def find(lib: Library) -> list[Path]:
        out = []
        for d in _characters(lib):
            if skip_training and _training_active(d):
                continue
            out += [d / n for n in names if (d / n).exists()]
        return out
    return find


def _unused_models(lib: Library) -> list[Path]:
    out = []
    for d in _characters(lib):
        models = d / "models"
        if not models.is_dir() or _training_active(d):
            continue
        try:
            cfg = json.loads((d / "config.json").read_text())
        except Exception:
            continue
        active = str(cfg.get("piper_model") or "").replace("\\", "/")
        out += [m for m in sorted(models.iterdir())
                if m.is_dir() and f"/models/{m.name}/" not in active]
    return out


def _stale_imports(lib: Library) -> list[Path]:
    if not lib.characters:
        return []
    return [d for d in lib.characters.iterdir() if d.is_dir() and d.name.startswith("_stage_")]


def _misc(*names: str):
    return lambda lib: [lib.misc / n for n in names if lib.misc and (lib.misc / n).exists()]


def _misc_loose(lib: Library) -> list[Path]:
    return [p for p in lib.misc.glob("*.wav")] if lib.misc else []


def _foley(lib: Library) -> list[Path]:
    return [p for p in lib.foley.iterdir()] if lib.foley else []


# key, label, description, finder, default-ticked
CATEGORIES: list[tuple[str, str, str, Callable[[Library], list[Path]], bool]] = [
    ("history",   "Generated clip history",      "Every clip generated for each character (favorites are kept)", _per_char("history"), True),
    ("model_try", "Model audition renders",      "Test lines rendered from the MODELS tab",                      _per_char("model_try", "piper_preview_long.wav", "piper_ref_sample.wav"), True),
    ("tmp",       "Scratch files",               "Import staging, edit previews, qualify previews",              _misc("_tmp"), True),
    ("candidates", "Unchosen checkpoints",       "Exported audition checkpoints not saved as models",            _per_char("piper_candidates", skip_training=True), True),
    ("trash",     "Reference trash",             "Clips you deleted while qualifying",                           _per_char("refs_trash"), False),
    ("misc",      "Misc generations & uploads",  "Clips and uploads that belong to no character, TTS, mixes",    lambda lib: _misc("uploads", "mixes", "base_voice_samples", "_experiments")(lib) + _misc_loose(lib), False),
    ("foley",     "Foley / sound effects",       "Everything in the foley folder",                               _foley, False),
    ("old_models", "Unused saved models",        "Saved Piper models other than each character's active one",    _unused_models, False),
    ("stale",     "Unfinished imports",          "Half-finished split & qualify staging characters",             _stale_imports, False),
    ("training",  "Piper training workspace",    "Training dataset and checkpoints — training can no longer be resumed", _per_char("piper_train", "piper_dataset", skip_training=True), False),
    ("originals", "Original reference backups",  "Pre-qualify copies of reference clips (refs_original)",        _per_char("refs_original"), False),
]


def _size(p: Path) -> int:
    try:
        if p.is_symlink() or p.is_file():
            return p.lstat().st_size
        total = 0
        for dp, _, files in os.walk(p):
            for f in files:
                try:
                    total += os.lstat(os.path.join(dp, f)).st_size
                except OSError:
                    pass
        return total
    except OSError:
        return 0


def scan(root: Path | str) -> list[dict]:
    lib = open_library(root)
    out = []
    for key, label, desc, find, default in CATEGORIES:
        paths = find(lib)
        out.append({"key": key, "label": label, "desc": desc, "default": default,
                    "items": len(paths), "bytes": sum(_size(p) for p in paths)})
    return out


def clean(root: Path | str, keys: list[str]) -> dict:
    """Delete the chosen categories. Returns {"freed": bytes, "removed": count}."""
    lib = open_library(root)
    freed = removed = 0
    for key, _, _, find, _ in CATEGORIES:
        if key not in keys:
            continue
        for p in find(lib):
            size = _size(p)
            try:
                if p.is_dir() and not p.is_symlink():
                    shutil.rmtree(p)
                else:
                    p.unlink()
            except OSError:
                continue
            freed += size
            removed += 1
    return {"freed": freed, "removed": removed}


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
