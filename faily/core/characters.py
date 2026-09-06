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


def get_ref_chain(name: str, include_excluded: bool = False) -> list[dict]:
    """Walk ancestry root→name; return [{audio, transcript, owner, file_key}] for
    all ref audio in each node.

    ref_clips entries flagged excluded=True are skipped by default — this is the single
    choke point every generation/training/count call site goes through, so opting a clip
    out here silently propagates everywhere. Pass include_excluded=True to see everything
    (used by the CHARACTERS tab's reference list, which shows excluded clips dimmed so
    they can be toggled back on).

    owner/file_key identify which character's config actually holds the clip (it may be
    an ancestor's, for an inherited sub-character chain) and its ref_clips file key, so
    callers can act on it in place (exclude/split) — file_key is None for a node's
    primary ref_audio, which isn't a ref_clips entry and can't be removed/split that way.
    """
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
                chain.append({"audio": audio, "transcript": char.get("transcript", ""), "owner": node, "file_key": None})
        for rc in char.get("ref_clips", []):
            if rc.get("excluded") and not include_excluded:
                continue
            audio = CHARACTERS_DIR / node / rc["file"]
            if audio.exists():
                chain.append({"audio": audio, "transcript": rc.get("transcript", ""), "owner": node, "file_key": rc["file"]})
    return chain


@contextmanager
def build_ref_audio(name: str, require_transcript: bool = False, max_duration: float | None = None):
    """Yield (Path, transcript) for the full ref chain.

    Single-entry chains return the existing file directly (no copy).
    Multi-entry chains are concatenated into a temp file that is deleted on exit.
    require_transcript=True: drop clips without transcripts so audio/text stay aligned (F5-TTS).
    max_duration: stop accumulating clips once this total duration (seconds) would be exceeded.
      F5-TTS generates [ref]+[gen] as one sequence and trims by duration — optimal ref is
      5–15 s; longer refs break the trim and cause the ref speech to bleed into output.
    """
    chain = get_ref_chain(name)
    if require_transcript:
        chain = [n for n in chain if n["transcript"].strip()]
    if max_duration is not None:
        import soundfile as _sf
        capped, total = [], 0.0
        for node in chain:
            dur = _sf.info(str(node["audio"])).duration
            if capped and total + dur > max_duration:
                break
            capped.append(node)
            total += dur
        chain = capped
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
        if data.ndim > 1:
            data = data.mean(axis=1)
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


def clip_quality_issues(path: Path) -> list[str]:
    """Lightweight heuristic checks on a reference clip — short/quiet/clipping/noisy.
    Returns a list of human-readable issue strings, empty if the clip looks fine.
    Not a substitute for actually listening, just enough to flag obvious problems
    before they quietly drag down cloning/training quality."""
    import numpy as np
    import soundfile as sf

    try:
        data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    except Exception:
        return ["couldn't read audio"]
    if data.ndim > 1:
        data = data.mean(axis=1)
    if len(data) == 0 or sr == 0:
        return ["empty audio"]

    issues = []
    duration = len(data) / sr
    if duration < 1.0:
        issues.append(f"very short ({duration:.1f}s)")
    peak = float(np.max(np.abs(data)))
    rms = float(np.sqrt(np.mean(data ** 2)))
    if peak >= 0.99:
        issues.append("clipping")
    if rms < 0.01:
        issues.append("very quiet")

    # SNR estimate: quietest short window in the clip stands in for the noise
    # floor (same trick used by the EDIT tab's denoiser, see modules/edit.py
    # _denoise_channel) vs. the clip's overall RMS as the signal level.
    win = max(int(0.2 * sr), 1)
    if len(data) > win * 3 and rms > 0:
        hop = max(win // 2, 1)
        n_wins = max((len(data) - win) // hop, 1)
        energies = [
            float(np.sqrt(np.mean(data[i * hop: i * hop + win] ** 2)))
            for i in range(n_wins)
        ]
        noise_floor = max(min(energies), 1e-6)
        snr_db = 20 * np.log10(rms / noise_floor)
        if snr_db < 15.0:
            issues.append(f"noisy (~{snr_db:.0f}dB SNR)")
    return issues


def add_ref_clip(name: str, clip_path: Path, transcript: str = "", source: str = "", category: str = "") -> Path:
    """Copy a generated clip into the character's ref pool and register it in config.

    source/category are optional provenance tags (e.g. source="buffer" for zero-shot
    padding clips from the buffer training wizard) — they persist permanently on the
    ref_clips entry so downstream code (get_ref_chain, training) doesn't need to
    regenerate or re-derive them, and the UI can group clips by source.
    """
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
    entry = {"file": f"refs/{dest.name}", "transcript": transcript}
    if source:
        entry["source"] = source
    if category:
        entry["category"] = category
    cfg.setdefault("ref_clips", []).append(entry)
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


def rename_ref_audio(name: str, new_stem: str) -> Path:
    """Rename the primary ref_audio file and update config."""
    p = _cfg(name)
    cfg = json.loads(p.read_text())
    old_file = cfg.get("ref_audio", "")
    if not old_file:
        raise ValueError("Character has no ref_audio")
    old_path = CHARACTERS_DIR / name / old_file
    if not old_path.exists():
        raise FileNotFoundError(f"ref_audio not found: {old_path}")
    new_file = new_stem + old_path.suffix
    new_path = old_path.parent / new_file
    if new_path.exists() and new_path != old_path:
        raise FileExistsError(f"'{new_file}' already exists")
    old_path.rename(new_path)
    cfg["ref_audio"] = new_file
    p.write_text(json.dumps(cfg, indent=2))
    return new_path


def set_ref_clip_excluded(name: str, file_key: str, excluded: bool) -> dict:
    """Opt a ref clip in/out of the character's VC reference chain without deleting it.
    All clips are included by default; this only ever narrows that default."""
    p = _cfg(name)
    cfg = json.loads(p.read_text())
    for rc in cfg.get("ref_clips", []):
        if rc["file"] == file_key:
            rc["excluded"] = excluded
            break
    p.write_text(json.dumps(cfg, indent=2))
    return cfg


def _split_sentences(text: str) -> list[str]:
    import re
    text = " ".join(text.split())
    if not text:
        return []
    return [p for p in re.split(r'(?<=[.!?])\s+', text) if p]


def split_long_ref_clip(name: str, file_key: str, target_seconds: float) -> int:
    """Split an over-long ref_clips entry into shorter pieces near target_seconds,
    cutting at detected internal silence gaps and dividing its transcript across
    the pieces at sentence boundaries. Replaces the original entry with the new
    ones (source/category carried over, so a buffer-generated clip stays tagged
    as such). Returns the number of pieces created — 0 (original left untouched)
    if there's fewer than 2 sentences to split across, or no usable silence gap
    is found (e.g. one long unbroken sentence can't be cut without breaking
    mid-word, so it's better left to the caller to exclude instead).
    """
    import soundfile as _sf
    from faily.modules.edit import find_silence_splits

    cfg = get_character(name)
    if not cfg:
        raise FileNotFoundError(f"Character '{name}' not found")
    entry = next((rc for rc in cfg.get("ref_clips", []) if rc["file"] == file_key), None)
    if entry is None:
        raise ValueError(f"Ref clip '{file_key}' not found in '{name}'")

    sentences = _split_sentences(entry.get("transcript", ""))
    if len(sentences) < 2:
        return 0

    audio_path = CHARACTERS_DIR / name / file_key
    data, sr = _sf.read(str(audio_path), dtype="float32", always_2d=False)
    duration = len(data) / sr if sr else 0.0
    desired_pieces = max(1, min(len(sentences), round(duration / target_seconds))) if target_seconds > 0 else 1
    if desired_pieces < 2:
        return 0

    split_points = find_silence_splits(data, sr, desired_pieces - 1)
    n_pieces = len(split_points) + 1
    if n_pieces < 2:
        return 0

    bounds = [0] + split_points + [len(data)]
    groups: list[list[str]] = [[] for _ in range(n_pieces)]
    for i, sent in enumerate(sentences):
        groups[min(i * n_pieces // len(sentences), n_pieces - 1)].append(sent)

    source = entry.get("source", "")
    category = entry.get("category", "")
    tmp_dir = Path(tempfile.mkdtemp(prefix="faily_split_"))
    created = 0
    try:
        for i in range(n_pieces):
            seg = data[bounds[i]:bounds[i + 1]]
            text = " ".join(groups[i]).strip()
            if not text or len(seg) == 0:
                continue
            tmp_path = tmp_dir / f"piece_{i:02d}.wav"
            _sf.write(str(tmp_path), seg, sr)
            add_ref_clip(name, tmp_path, transcript=text, source=source, category=category)
            created += 1
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    if created:
        remove_ref_clip(name, file_key)
    return created


def update_ref_clip(name: str, file_key: str, new_stem: str | None = None, new_transcript: str | None = None) -> dict:
    """Update a ref_clip's filename and/or transcript in-place."""
    p = _cfg(name)
    cfg = json.loads(p.read_text())
    for rc in cfg.get("ref_clips", []):
        if rc["file"] == file_key:
            if new_transcript is not None:
                rc["transcript"] = new_transcript
            if new_stem is not None:
                old_path = CHARACTERS_DIR / name / file_key
                prefix = file_key.rsplit("/", 1)[0] + "/" if "/" in file_key else ""
                new_rel = prefix + new_stem + old_path.suffix
                new_path = CHARACTERS_DIR / name / new_rel
                if new_path.exists() and new_path != old_path:
                    raise FileExistsError(f"'{new_stem}' already exists")
                old_path.rename(new_path)
                rc["file"] = new_rel
            break
    else:
        raise ValueError(f"Ref clip '{file_key}' not found in '{name}'")
    p.write_text(json.dumps(cfg, indent=2))
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


def set_piper_model(name: str, model_path: str) -> dict:
    """Store the trained Piper .onnx model path in the character config."""
    p = _cfg(name)
    if not p.exists():
        raise FileNotFoundError(f"Character '{name}' not found")
    cfg = json.loads(p.read_text())
    cfg["piper_model"] = model_path
    p.write_text(json.dumps(cfg, indent=2))
    return cfg
