import json
import shutil
from datetime import datetime
from pathlib import Path

CHARACTERS_DIR = Path("outputs/characters")


def _cfg(name: str) -> Path:
    return CHARACTERS_DIR / name / "config.json"


def list_characters() -> list[dict]:
    if not CHARACTERS_DIR.exists():
        return []
    result = []
    for d in sorted(CHARACTERS_DIR.iterdir()):
        p = d / "config.json"
        if d.is_dir() and p.exists():
            try:
                result.append(json.loads(p.read_text()))
            except Exception:
                pass
    return result


def get_character(name: str) -> dict | None:
    p = _cfg(name)
    return json.loads(p.read_text()) if p.exists() else None


def get_ref_path(name: str) -> Path | None:
    """Resolve the reference audio path, following parent link for sub-characters."""
    char = get_character(name)
    if not char:
        return None
    source = char.get("parent") or name
    base = get_character(source)
    if not base or "ref_audio" not in base:
        return None
    return CHARACTERS_DIR / source / base["ref_audio"]


def save_character(name: str, ref_path: Path, transcript: str = "") -> dict:
    """Create or overwrite a base character from a reference audio file."""
    char_dir = CHARACTERS_DIR / name
    char_dir.mkdir(parents=True, exist_ok=True)
    dest = char_dir / ("ref" + ref_path.suffix)
    shutil.copy2(str(ref_path), str(dest))
    cfg = {
        "name": name,
        "ref_audio": dest.name,
        "transcript": transcript,
        "created": datetime.now().isoformat(),
    }
    _cfg(name).write_text(json.dumps(cfg, indent=2))
    return cfg


def save_sub_character(
    name: str,
    parent: str,
    backend: str,
    param1: float,
    param2: float,
    speed: float = 1.0,
    style_prompt: str = "",
) -> dict:
    """Save an expression variant of an existing character."""
    (CHARACTERS_DIR / name).mkdir(parents=True, exist_ok=True)
    cfg = {
        "name": name,
        "parent": parent,
        "backend": backend,
        "param1": param1,
        "param2": param2,
        "speed": speed,
        "style_prompt": style_prompt,
        "created": datetime.now().isoformat(),
    }
    _cfg(name).write_text(json.dumps(cfg, indent=2))
    return cfg


def delete_character(name: str):
    char_dir = CHARACTERS_DIR / name
    if char_dir.exists():
        shutil.rmtree(str(char_dir))


def add_clip_to_character(name: str, clip_path: Path) -> Path:
    """Copy a generated clip into the character's clip collection."""
    dest_dir = CHARACTERS_DIR / name / "clips"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / clip_path.name
    shutil.copy2(str(clip_path), str(dest))
    return dest


def add_clip_to_favorites(name: str, clip_path: Path) -> Path:
    """Copy a generated clip into the character's favorites folder."""
    dest_dir = CHARACTERS_DIR / name / "favorites"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / clip_path.name
    shutil.copy2(str(clip_path), str(dest))
    return dest


def list_character_favorites(name: str) -> list[Path]:
    fav_dir = CHARACTERS_DIR / name / "favorites"
    if not fav_dir.exists():
        return []
    return sorted(fav_dir.glob("*.wav"), reverse=True)


def update_character_metadata(name: str, updates: dict) -> dict:
    """Update specific text fields in a character config. Name/created/ref_audio are protected."""
    p = _cfg(name)
    if not p.exists():
        raise FileNotFoundError(f"Character '{name}' not found")
    cfg = json.loads(p.read_text())
    _safe = {"transcript", "style_prompt", "speed"}
    for k, v in updates.items():
        if k in _safe:
            cfg[k] = v
    p.write_text(json.dumps(cfg, indent=2))
    return cfg
