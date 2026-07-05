import json
import shutil
import tempfile
from contextlib import contextmanager
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
    """Walk ancestry root→name; return [{audio, transcript}] for all ref audio in each node."""
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
        node = char["name"]
        if "ref_audio" in char:
            audio = CHARACTERS_DIR / node / char["ref_audio"]
            if audio.exists():
                chain.append({"audio": audio, "transcript": char.get("transcript", "")})
        for rc in char.get("ref_clips", []):
            audio = CHARACTERS_DIR / node / rc["file"]
            if audio.exists():
                chain.append({"audio": audio, "transcript": rc.get("transcript", "")})
    return chain


@contextmanager
def build_ref_audio(name: str):
    """Yield (Path, transcript) for the full ref chain.

    Single-entry chains return the existing file directly (no copy).
    Multi-entry chains are concatenated into a temp file that is deleted on exit.
    """
    chain = get_ref_chain(name)
    if not chain:
        yield None, ""
        return
    transcript = " ".join(n["transcript"] for n in chain if n["transcript"]).strip()
    if len(chain) == 1:
        yield chain[0]["audio"], transcript
        return
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
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        tmp = Path(f.name)
    try:
        sf.write(str(tmp), combined, target_sr)
        yield tmp, transcript
    finally:
        if tmp.exists():
            tmp.unlink()


def get_ref_path(name: str) -> Path | None:
    """Return the first ref audio in the chain, for display/preview only.

    Use build_ref_audio() at generation time to get the full concatenated chain.
    """
    chain = get_ref_chain(name)
    return chain[0]["audio"] if chain else None


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


def add_ref_clip(name: str, clip_path: Path, transcript: str = "") -> Path:
    """Copy a generated clip into the character's ref pool and register it in config."""
    char_dir = CHARACTERS_DIR / name
    if not (char_dir / "config.json").exists():
        raise FileNotFoundError(f"Character '{name}' not found")
    refs_dir = char_dir / "refs"
    refs_dir.mkdir(exist_ok=True)
    i = 1
    while True:
        dest = refs_dir / f"ref_{i:03d}{clip_path.suffix}"
        if not dest.exists():
            break
        i += 1
    shutil.copy2(str(clip_path), str(dest))
    cfg = json.loads(_cfg(name).read_text())
    cfg.setdefault("ref_clips", []).append({"file": f"refs/{dest.name}", "transcript": transcript})
    _cfg(name).write_text(json.dumps(cfg, indent=2))
    return dest


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


def remove_ref_clip(name: str, file_key: str) -> dict:
    """Remove a ref clip from ref_clips by its file key and delete the file."""
    p = _cfg(name)
    if not p.exists():
        raise FileNotFoundError(f"Character '{name}' not found")
    cfg = json.loads(p.read_text())
    cfg["ref_clips"] = [rc for rc in cfg.get("ref_clips", []) if rc["file"] != file_key]
    p.write_text(json.dumps(cfg, indent=2))
    audio = CHARACTERS_DIR / name / file_key
    if audio.exists():
        audio.unlink()
    return cfg


def set_rvc_model(name: str, model_path: str) -> dict:
    """Store the trained RVC model path in the character config."""
    p = _cfg(name)
    if not p.exists():
        raise FileNotFoundError(f"Character '{name}' not found")
    cfg = json.loads(p.read_text())
    cfg["rvc_model"] = model_path
    p.write_text(json.dumps(cfg, indent=2))
    return cfg
