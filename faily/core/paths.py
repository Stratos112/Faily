"""Where everything under outputs/ lives.

outputs/
  characters/<name>/   everything about one character: references, inflated clips,
                       trained models, favorites, and history/ (every clip generated
                       for them, zero-shot or Piper)
  foley/               sound effects (no character connection)
  misc/                audio that belongs to no character: uploads never used as a
                       character basis, zero-shot clones never saved as a character,
                       TTS, mixes. misc/_tmp holds scratch files and is safe to delete.
"""
import re
import shutil
from pathlib import Path

OUTPUTS = Path("outputs")
CHARACTERS_DIR = OUTPUTS / "characters"
FOLEY_DIR = OUTPUTS / "foley"
MISC_DIR = OUTPUTS / "misc"
UPLOADS_DIR = MISC_DIR / "uploads"
TMP_DIR = MISC_DIR / "_tmp"
MIX_DIR = MISC_DIR / "mixes"


def _is_character(name: str | None) -> bool:
    return bool(name) and not name.startswith("_") and (CHARACTERS_DIR / name / "config.json").exists()


def owner_of(path: Path | str | None) -> str | None:
    """The character whose folder holds `path`, if any."""
    if not path:
        return None
    try:
        parts = Path(path).resolve().relative_to(CHARACTERS_DIR.resolve()).parts
    except ValueError:
        return None
    return parts[0] if parts and _is_character(parts[0]) else None


def history_dir(char_name: str | None = None, ref_path: Path | str | None = None) -> Path:
    """Folder for a freshly generated clip: the character's history if the clip is
    for a saved character (named, or cloned from one of their clips), else misc."""
    owner = char_name if _is_character(char_name) else owner_of(ref_path)
    return CHARACTERS_DIR / owner / "history" if owner else MISC_DIR


def ensure_layout() -> None:
    for d in (CHARACTERS_DIR, FOLEY_DIR, MISC_DIR):
        d.mkdir(parents=True, exist_ok=True)


# ── one-time move from the old layout (vc/ tts/ sfx/ edit/ mix/ …) ──────────────

def _slug(name: str) -> str:
    # same slug vc._make_clip_name puts at the front of a clip's filename
    return re.sub(r"[^a-z0-9]+", "_", name.lower().strip()).strip("_")[:16]


def _owner_from_filename(stem: str, names: list[str]) -> str | None:
    best, best_len = None, 0
    for n in names:
        for prefix in (f"{_slug(n)}_", f"piper_{n}_", f"speak_{n}_"):
            if len(prefix) > 1 and stem.startswith(prefix) and len(prefix) > best_len:
                best, best_len = n, len(prefix)
    return best


def _move(src: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src.name
    i = 1
    while dest.exists():
        dest = dest_dir / f"{src.stem}_{i}{src.suffix}"
        i += 1
    shutil.move(str(src), str(dest))


def _move_all(src_dir: Path, dest_dir: Path) -> None:
    if src_dir.is_dir():
        for p in sorted(src_dir.iterdir()):
            _move(p, dest_dir)


def _rmdir_quiet(d: Path) -> None:
    try:
        d.rmdir()
    except OSError:
        pass


def migrate_legacy_layout() -> int:
    """Move files from the old outputs/{vc,tts,sfx,edit,mix,speak,...} folders into
    the characters/foley/misc layout. Returns how many files/folders were moved.
    Idempotent: does nothing once the old folders are gone."""
    ensure_layout()
    names = [d.name for d in CHARACTERS_DIR.iterdir() if _is_character(d.name)]
    moved = 0

    def count(n=1):
        nonlocal moved
        moved += n

    vc = OUTPUTS / "vc"
    if vc.is_dir():
        for p in sorted(vc.iterdir()):
            if p.is_file():
                owner = _owner_from_filename(p.stem, names)
                _move(p, CHARACTERS_DIR / owner / "history" if owner else MISC_DIR)
            elif p.name == "refs":
                _move_all(p, UPLOADS_DIR)
                _rmdir_quiet(p)
            else:
                shutil.move(str(p), str(_free(TMP_DIR / p.name)))     # stale staging folders
            count()
        _rmdir_quiet(vc)

    for old, new in (("tts", MISC_DIR), ("sfx", FOLEY_DIR), ("mix", MIX_DIR),
                     ("base_voice_samples", MISC_DIR / "base_voice_samples")):
        d = OUTPUTS / old
        if d.is_dir():
            n = sum(1 for _ in d.iterdir())
            _move_all(d, new)
            count(n)
            _rmdir_quiet(d)

    for old in ("edit", "_qualify_preview", "speak"):
        d = OUTPUTS / old
        if d.is_dir():
            n = sum(1 for _ in d.iterdir())
            _move_all(d, TMP_DIR / old)
            count(n)
            _rmdir_quiet(d)

    # audition renders from the MODELS tab: <character>_<model id>.wav
    tried = OUTPUTS / "_model_try"
    if tried.is_dir():
        for p in sorted(tried.iterdir()):
            owner = next((n for n in sorted(names, key=len, reverse=True) if p.stem.startswith(f"{n}_")), None)
            _move(p, CHARACTERS_DIR / owner / "model_try" if owner else TMP_DIR / "model_try")
            count()
        _rmdir_quiet(tried)

    exp = OUTPUTS / "_experiments"
    if exp.is_dir():
        shutil.move(str(exp), str(_free(MISC_DIR / "_experiments")))
        count()

    return moved


def _free(p: Path) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    i = 1
    cand = p
    while cand.exists():
        cand = p.with_name(f"{p.name}_{i}")
        i += 1
    return cand
