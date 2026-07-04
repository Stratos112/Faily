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


def get_ref_chain(name: str) -> list[dict]:
    """Walk ancestry root→name; return [{audio, transcript}] for nodes that have audio."""
    path_up, seen, current = [], set(), name
    while current and current not in seen:
        seen.add(current)
        char = get_character(current)
        if not char:
            break
        path_up.append(char)
        current = char.get("parent")
    chain = []
    for char in reversed(path_up):
        if "ref_audio" not in char:
            continue
        audio = CHARACTERS_DIR / char["name"] / char["ref_audio"]
        if audio.exists():
            chain.append({"audio": audio, "transcript": char.get("transcript", "")})
    return chain


def concat_ref_audio(name: str) -> tuple[Path | None, str]:
    """Return (ref_path, transcript) for the full ancestry chain, concatenating if needed.

    Writes _ref_concat.wav into the character dir when multiple files need merging.
    """
    chain = get_ref_chain(name)
    if not chain:
        return None, ""
    transcript = " ".join(n["transcript"] for n in chain if n["transcript"]).strip()
    if len(chain) == 1:
        return chain[0]["audio"], transcript
    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio
    arrays, target_sr = [], None
    for node in chain:
        data, sr = sf.read(str(node["audio"]), dtype="float32", always_2d=False)
        if target_sr is None:
            target_sr = sr
        elif sr != target_sr:
            wav = torch.from_numpy(data).unsqueeze(0)
            data = torchaudio.functional.resample(wav, sr, target_sr).squeeze(0).numpy()
        arrays.append(data)
    combined = np.concatenate(arrays)
    out = CHARACTERS_DIR / name / "_ref_concat.wav"
    sf.write(str(out), combined, target_sr)
    return out, transcript


def get_ref_path(name: str) -> Path | None:
    """Return the resolved reference audio path for name (concatenated chain if needed)."""
    path, _ = concat_ref_audio(name)
    return path


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
    ref_path: Path | None = None,
    transcript: str = "",
) -> dict:
    """Save an expression variant of an existing character."""
    char_dir = CHARACTERS_DIR / name
    char_dir.mkdir(parents=True, exist_ok=True)
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
    if ref_path is not None:
        dest = char_dir / ("ref" + ref_path.suffix)
        shutil.copy2(str(ref_path), str(dest))
        cfg["ref_audio"] = dest.name
        cfg["transcript"] = transcript
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


def list_character_clips(name: str) -> list[Path]:
    clips_dir = CHARACTERS_DIR / name / "clips"
    if not clips_dir.exists():
        return []
    return sorted(clips_dir.glob("*.wav"), reverse=True)


def list_character_favorites(name: str) -> list[Path]:
    fav_dir = CHARACTERS_DIR / name / "favorites"
    if not fav_dir.exists():
        return []
    return sorted(fav_dir.glob("*.wav"), reverse=True)


def rename_character_file(char_name: str, subfolder: str, old_name: str, new_name: str) -> Path:
    """Rename a clip inside a character's clips/ or favorites/ subfolder."""
    if not new_name.lower().endswith(".wav"):
        new_name += ".wav"
    src = CHARACTERS_DIR / char_name / subfolder / old_name
    if not src.exists():
        raise FileNotFoundError(f"File not found: {src}")
    dest = src.parent / new_name
    if dest.exists() and dest != src:
        raise FileExistsError(f"'{new_name}' already exists")
    src.rename(dest)
    return dest


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
