"""Piper TTS training and inference via isolated subprocess."""
import asyncio
import csv
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import scipy.signal as sps
import soundfile as sf

_PROJECT_ROOT = Path(__file__).parent.parent.parent
PIPER_VENV    = _PROJECT_ROOT / "piper_venv"
PIPER_CKPTS   = _PROJECT_ROOT / "piper_checkpoints"
_SR           = 22050
_IS_WIN       = sys.platform == "win32"


# ── base voices ─────────────────────────────────────────────────────────────
# Curated subset of rhasspy's published English voices — verified (via the HF
# API, not assumed) to have both a resumable Lightning checkpoint
# (rhasspy/piper-checkpoints) and a matching inference config
# (rhasspy/piper-voices) in the "medium" quality tier. Not every voice rhasspy
# publishes has both — some directories only have the exported .onnx, not the
# training checkpoint needed to resume fine-tuning from.
#
# "lessac" is special: it's the one setup_piper.bat already downloads
# unconditionally into piper_checkpoints/ (flat, no subfolder) — kept exactly
# as-is so existing installs need no migration. Every other voice downloads
# on demand into its own piper_checkpoints/{key}/ subfolder the first time
# it's actually selected for training.
BASE_VOICES: dict[str, dict] = {
    "lessac": {
        "label": "Lessac (default)",
        "desc": "Neutral American female voice — Faily's original default.",
        "ckpt_url": "https://huggingface.co/datasets/rhasspy/piper-checkpoints/resolve/main/en/en_US/lessac/medium/epoch=2164-step=1355540.ckpt",
        "cfg_url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json",
        "onnx_url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx",
        "ckpt_filename": "epoch=2164-step=1355540.ckpt",
        "cfg_filename": "en_US-lessac-medium.onnx.json",
    },
    "ryan": {
        "label": "Ryan",
        "desc": "Deeper American male voice.",
        "ckpt_url": "https://huggingface.co/datasets/rhasspy/piper-checkpoints/resolve/main/en/en_US/ryan/medium/epoch=4641-step=3104302.ckpt",
        "cfg_url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/medium/en_US-ryan-medium.onnx.json",
        "onnx_url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/medium/en_US-ryan-medium.onnx",
        "ckpt_filename": "epoch=4641-step=3104302.ckpt",
        "cfg_filename": "en_US-ryan-medium.onnx.json",
    },
    "amy": {
        "label": "Amy",
        "desc": "American female voice, different timbre from Lessac.",
        "ckpt_url": "https://huggingface.co/datasets/rhasspy/piper-checkpoints/resolve/main/en/en_US/amy/medium/epoch=6679-step=1554200.ckpt",
        "cfg_url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx.json",
        "onnx_url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx",
        "ckpt_filename": "epoch=6679-step=1554200.ckpt",
        "cfg_filename": "en_US-amy-medium.onnx.json",
    },
    "hfc_male": {
        "label": "HFC Male",
        "desc": "Home Assistant community male voice.",
        "ckpt_url": "https://huggingface.co/datasets/rhasspy/piper-checkpoints/resolve/main/en/en_US/hfc_male/medium/epoch=2785-step=2128064.ckpt",
        "cfg_url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/hfc_male/medium/en_US-hfc_male-medium.onnx.json",
        "onnx_url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/hfc_male/medium/en_US-hfc_male-medium.onnx",
        "ckpt_filename": "epoch=2785-step=2128064.ckpt",
        "cfg_filename": "en_US-hfc_male-medium.onnx.json",
    },
    "hfc_female": {
        "label": "HFC Female",
        "desc": "Home Assistant community female voice.",
        "ckpt_url": "https://huggingface.co/datasets/rhasspy/piper-checkpoints/resolve/main/en/en_US/hfc_female/medium/epoch=2868-step=1575188.ckpt",
        "cfg_url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/hfc_female/medium/en_US-hfc_female-medium.onnx.json",
        "onnx_url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/hfc_female/medium/en_US-hfc_female-medium.onnx",
        "ckpt_filename": "epoch=2868-step=1575188.ckpt",
        "cfg_filename": "en_US-hfc_female-medium.onnx.json",
    },
}

SAMPLE_TEXT = "The quick brown fox jumps over the lazy dog."
# Relative, like CHARACTERS_DIR etc. — the app always runs with the project
# root as cwd, and UI code does path.relative_to(Path("outputs")) to build
# /outputs/... URLs, which requires this to stay relative too.
_BASE_VOICE_SAMPLES_DIR = Path("outputs/base_voice_samples")


def base_voice_dir(key: str) -> Path:
    return PIPER_CKPTS if key == "lessac" else PIPER_CKPTS / key


def base_voice_ready(key: str) -> bool:
    """True if this base voice's checkpoint + config are already on disk."""
    if key not in BASE_VOICES:
        return False
    d = base_voice_dir(key)
    return bool(list(d.glob("*.ckpt"))) and bool(list(d.glob("*.onnx.json"))) if d.exists() else False


def base_voice_paths(key: str) -> tuple[Path, Path]:
    """Returns (ckpt_path, cfg_path) for an already-downloaded base voice."""
    d = base_voice_dir(key)
    ckpts = sorted(d.glob("*.ckpt")) if d.exists() else []
    cfgs = sorted(d.glob("*.onnx.json")) if d.exists() else []
    if not ckpts or not cfgs:
        raise FileNotFoundError(f"Base voice '{key}' isn't downloaded yet")
    return ckpts[-1], cfgs[-1]


def _sample_onnx_filename(key: str) -> str:
    cfg_filename = BASE_VOICES[key]["cfg_filename"]
    return cfg_filename[: -len(".json")]


def sample_ready(key: str) -> bool:
    """True if this base voice's small preview .onnx (+ matching config) is
    already on disk — independent of whether the (much larger) Lightning
    training checkpoint has been downloaded."""
    if key not in BASE_VOICES:
        return False
    d = base_voice_dir(key)
    return (d / _sample_onnx_filename(key)).exists() and (d / BASE_VOICES[key]["cfg_filename"]).exists()


def sample_onnx_path(key: str) -> Path:
    return base_voice_dir(key) / _sample_onnx_filename(key)


def _download_with_progress(url: str, dest: Path, label: str, log_cb) -> None:
    import urllib.request
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        last_pct = -1
        with open(tmp, "wb") as f:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = int(downloaded * 100 / total)
                    if pct != last_pct and pct % 10 == 0:
                        log_cb(f"  {label}: {pct}% ({downloaded // (1024*1024)}/{total // (1024*1024)} MB)")
                        last_pct = pct
    tmp.rename(dest)


async def download_base_voice(key: str, log_cb) -> None:
    """Download a base voice's checkpoint + config on demand (~800 MB total),
    skipping any files already present."""
    if key not in BASE_VOICES:
        raise ValueError(f"Unknown base voice: {key}")
    voice = BASE_VOICES[key]
    dest_dir = base_voice_dir(key)
    dest_dir.mkdir(parents=True, exist_ok=True)

    ckpt_dest = dest_dir / voice["ckpt_filename"]
    cfg_dest = dest_dir / voice["cfg_filename"]

    if not ckpt_dest.exists():
        log_cb(f"Downloading {voice['label']} checkpoint (~800 MB)…")
        await asyncio.to_thread(_download_with_progress, voice["ckpt_url"], ckpt_dest, ckpt_dest.name, log_cb)
    else:
        log_cb(f"{voice['label']} checkpoint already present")

    if not cfg_dest.exists():
        log_cb(f"Downloading {voice['label']} config…")
        await asyncio.to_thread(_download_with_progress, voice["cfg_url"], cfg_dest, cfg_dest.name, log_cb)
    else:
        log_cb(f"{voice['label']} config already present")


async def download_sample_voice(key: str, log_cb) -> None:
    """Download just the small pre-exported .onnx + its config (tens of MB,
    not the ~800 MB Lightning checkpoint) so the base voice can be auditioned
    before committing to a full download for training."""
    if key not in BASE_VOICES:
        raise ValueError(f"Unknown base voice: {key}")
    voice = BASE_VOICES[key]
    dest_dir = base_voice_dir(key)
    dest_dir.mkdir(parents=True, exist_ok=True)

    cfg_dest = dest_dir / voice["cfg_filename"]
    if not cfg_dest.exists():
        log_cb(f"Downloading {voice['label']} config…")
        await asyncio.to_thread(_download_with_progress, voice["cfg_url"], cfg_dest, cfg_dest.name, log_cb)

    onnx_dest = dest_dir / _sample_onnx_filename(key)
    if not onnx_dest.exists():
        log_cb(f"Downloading {voice['label']} preview model…")
        await asyncio.to_thread(_download_with_progress, voice["onnx_url"], onnx_dest, onnx_dest.name, log_cb)


async def generate_base_voice_sample(key: str, log_cb=None) -> Path:
    """Ensure the small preview model is downloaded, then synthesize a short
    fixed test line with it. Cached on disk — later calls for the same voice
    return instantly instead of re-running inference."""
    if key not in BASE_VOICES:
        raise ValueError(f"Unknown base voice: {key}")
    out_path = _BASE_VOICE_SAMPLES_DIR / f"{key}.wav"
    if out_path.exists():
        return out_path
    if not can_infer():
        raise RuntimeError("Piper binary not found in piper_venv — run the setup script first.")
    if not sample_ready(key):
        await download_sample_voice(key, log_cb or (lambda _line: None))
    _BASE_VOICE_SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    await asyncio.to_thread(infer, SAMPLE_TEXT, sample_onnx_path(key), out_path)
    return out_path


async def generate_character_ref_sample(model_path: Path) -> Path:
    """Synthesize SAMPLE_TEXT with a character's trained Piper model, for use as a
    clean, single-take voice-conversion reference in EXPRESSION (Piper's own output
    is denoised, consistent studio audio — a cleaner VC target than noisier uploads).
    Cached next to the model; regenerated automatically if the model is retrained."""
    out_path = model_path.with_name("piper_ref_sample.wav")
    if out_path.exists() and out_path.stat().st_mtime >= model_path.stat().st_mtime:
        return out_path
    if not can_infer():
        raise RuntimeError("Piper binary not found in piper_venv — run the setup script first.")
    await asyncio.to_thread(infer, SAMPLE_TEXT, model_path, out_path)
    return out_path


async def _read_checkpoint_epoch(py: str, ckpt_path: Path) -> int:
    """Read the actual current_epoch from inside a Lightning checkpoint —
    more reliable than parsing it out of the filename, which isn't
    consistently formatted across different base voices (some use
    epoch=N-step=M.ckpt, others use arbitrary names like "bryce-3499.ckpt"
    with no "epoch=" substring at all)."""
    code = (
        "import torch, sys, pathlib\n"
        "if sys.platform == 'win32':\n"
        "    pathlib.PosixPath = pathlib.WindowsPath\n"
        f"ckpt = torch.load({str(ckpt_path)!r}, map_location='cpu', weights_only=False)\n"
        "print(ckpt.get('epoch', 0))\n"
    )
    r = await asyncio.to_thread(
        subprocess.run, [py, "-c", code], capture_output=True, timeout=120, text=True,
    )
    if r.returncode != 0:
        detail = (r.stderr or r.stdout or "").strip()
        raise RuntimeError(f"Couldn't read checkpoint epoch: {detail[-500:]}")
    return int(r.stdout.strip())


def _python() -> Path:
    return PIPER_VENV / ("Scripts/python.exe" if _IS_WIN else "bin/python")


def _piper_bin() -> Path:
    return PIPER_VENV / ("Scripts/piper.exe" if _IS_WIN else "bin/piper")


def can_infer() -> bool:
    return _piper_bin().exists()


def can_train() -> bool:
    """True if the piper venv has everything needed to start training."""
    return diagnose() is None


def diagnose() -> str | None:
    """Human-readable reason piper training isn't available, or None if ready."""
    py = _python()
    if not py.exists():
        return f"piper venv not found at {PIPER_VENV} — run the setup script."

    check = "phonemizer" if _IS_WIN else "piper_train"
    try:
        r = subprocess.run(
            [str(py), "-c", f"import {check}"],
            capture_output=True, timeout=15, text=True,
        )
    except Exception as exc:
        return f"couldn't run the piper venv's python ({exc}) — rerun the setup script."

    if r.returncode != 0:
        lines = (r.stderr or r.stdout or "").strip().splitlines()
        detail = lines[-1] if lines else f"exit code {r.returncode}"
        if "No Python at" in detail:
            return (
                f"the piper venv is broken ({detail}) — it was likely created with a "
                "different Python (e.g. from WSL). Delete piper_venv/ and rerun the "
                "setup script."
            )
        return f"{detail} — rerun the setup script."
    return None


def is_ready() -> bool:
    return can_infer()


# ── piper_phonemize shim ──────────────────────────────────────────────────────
# On Windows, piper-phonemize (C extension) can't be compiled without dev headers.
# We drop a pure-Python shim into the venv that satisfies piper-train's import
# using the `phonemizer` package (which calls the espeak-ng binary directly).

_SHIM_CODE = r'''
"""
piper_phonemize shim.
Replaces the C extension with phonemizer (calls espeak-ng binary).
Implements the same Python-level API as piper_phonemize 1.1.0's __init__.py
(see https://github.com/rhasspy/piper-phonemize/blob/v1.1.0/piper_phonemize/__init__.py)
so piper_train.preprocess's `from piper_phonemize import (...)` succeeds and
produces phoneme IDs compatible with the pretrained checkpoint's embeddings.
Auto-generated by Faily — do not edit.
"""
import json, os, shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ── locate espeak-ng binary ───────────────────────────────────────────────────
def _ensure_espeak_path():
    if shutil.which("espeak-ng"):
        return
    for candidate in [
        r"C:\Program Files\eSpeak NG",
        r"C:\Program Files (x86)\eSpeak NG",
    ]:
        if Path(candidate).is_dir():
            os.environ["PATH"] = candidate + os.pathsep + os.environ.get("PATH", "")
            return

_ensure_espeak_path()


# ── phoneme ID map (loaded from piper config JSON) — must match the base
# checkpoint's embedding table exactly, so we read it from the voice config
# rather than hardcoding it. ──────────────────────────────────────────────────
def _load_phoneme_id_map() -> Dict[str, List[int]]:
    here = Path(__file__).resolve()
    for parent in here.parents:
        cfgs = list((parent / "piper_checkpoints").glob("*.onnx.json"))
        if cfgs:
            try:
                data = json.loads(cfgs[0].read_text(encoding="utf-8"))
                if "phoneme_id_map" in data:
                    return data["phoneme_id_map"]
            except Exception:
                pass
    return {}

DEFAULT_PHONEME_ID_MAP: Dict[str, List[int]] = _load_phoneme_id_map()
_PAD, _BOS, _EOS = "_", "^", "$"


# ── IPA tokenizer ─────────────────────────────────────────────────────────────
def _tokenize(ipa: str) -> List[str]:
    """Greedily split IPA string into tokens present in the phoneme map."""
    keys = sorted(DEFAULT_PHONEME_ID_MAP, key=len, reverse=True)
    tokens, i = [], 0
    while i < len(ipa):
        for k in keys:
            if ipa[i : i + len(k)] == k:
                tokens.append(k)
                i += len(k)
                break
        else:
            i += 1
    return tokens


# ── main API ──────────────────────────────────────────────────────────────────
def phonemize_espeak(
    text: str,
    voice: str = "en-us",
    data_path=None,
) -> List[List[str]]:
    """
    Convert text to IPA phoneme token lists using phonemizer + espeak-ng.
    Matches the API of the real piper_phonemize C extension.
    """
    from phonemizer.backend import EspeakBackend
    from phonemizer.separator import Separator

    backend = EspeakBackend(
        voice,
        with_stress=True,
        language_switch="remove-flags",
        words_mismatch="ignore",
    )
    sep = Separator(phone=None, word=" ", syllable="")
    sentences = [s for s in text.split("\n") if s.strip()] or [text]
    ipa_list = backend.phonemize(sentences, separator=sep, strip=True)
    return [_tokenize(s) for s in ipa_list]


def phonemize_codepoints(text: str, casing: str = "fold") -> List[List[str]]:
    """Phonemize text as raw UTF-8 codepoints (--phoneme-type text mode)."""
    if casing == "lower":
        text = text.lower()
    elif casing == "upper":
        text = text.upper()
    elif casing == "fold":
        text = text.casefold()
    sentences = [s for s in text.split("\n") if s.strip()] or [text]
    return [list(s) for s in sentences]


def _phonemes_to_ids(phonemes: List[str]) -> Tuple[List[int], Dict[str, int]]:
    """Mirrors piper_phonemize's phonemes_to_ids: BOS, pad-interspersed
    phonemes (pad after every phoneme, including the last), then EOS."""
    ids: List[int] = []
    missing: Dict[str, int] = {}
    pad_ids = DEFAULT_PHONEME_ID_MAP.get(_PAD, [0])

    ids.extend(DEFAULT_PHONEME_ID_MAP.get(_BOS, [1]))
    ids.extend(pad_ids)
    for p in phonemes:
        mapped = DEFAULT_PHONEME_ID_MAP.get(p)
        if mapped is None:
            missing[p] = missing.get(p, 0) + 1
            continue
        ids.extend(mapped)
        ids.extend(pad_ids)
    ids.extend(DEFAULT_PHONEME_ID_MAP.get(_EOS, [2]))
    return ids, missing


def phoneme_ids_espeak(
    phonemes: List[str],
    missing_phonemes: Optional[object] = None,
) -> List[int]:
    ids, missing = _phonemes_to_ids(phonemes)
    if missing_phonemes is not None and missing:
        missing_phonemes.update(missing)
    return ids


def phoneme_ids_codepoints(
    language: str,
    phonemes: List[str],
    missing_phonemes: Optional[object] = None,
) -> List[int]:
    # Faily only trains with espeak-derived phonemes; codepoints mode isn't
    # exercised, but the symbol must exist for piper_train's import to succeed.
    ids, missing = _phonemes_to_ids(phonemes)
    if missing_phonemes is not None and missing:
        missing_phonemes.update(missing)
    return ids


def get_espeak_map() -> Dict[str, List[int]]:
    return DEFAULT_PHONEME_ID_MAP


def get_codepoints_map() -> Dict[str, Dict[str, List[int]]]:
    return {}


def get_max_phonemes() -> int:
    return 256


def tashkeel_run(text: str, tashkeel_model=None) -> str:
    # Arabic diacritization — not used for English training.
    return text
'''


def _site_packages() -> Path:
    """Return the site-packages directory inside the piper venv."""
    if _IS_WIN:
        return PIPER_VENV / "Lib" / "site-packages"
    lib = PIPER_VENV / "lib"
    py_dirs = sorted(lib.glob("python3.*"))
    if not py_dirs:
        raise RuntimeError("Cannot locate piper venv site-packages")
    return py_dirs[0] / "site-packages"


_PHONEMIZE_NAMES = (
    "phonemize_espeak", "phonemize_codepoints",
    "phoneme_ids_espeak", "phoneme_ids_codepoints",
    "get_codepoints_map", "get_espeak_map", "get_max_phonemes", "tashkeel_run",
)


def _ensure_phonemize_shim():
    """
    Write piper_phonemize shim into the venv if the real C extension isn't
    there, or if a previously-written shim is missing names piper_train
    actually imports (e.g. from an older version of this shim).
    Safe to call on Linux too — skips if the real package already satisfies it.
    """
    probe = "from piper_phonemize import " + ", ".join(_PHONEMIZE_NAMES)
    r = subprocess.run(
        [str(_python()), "-c", probe],
        capture_output=True, timeout=15,
    )
    if r.returncode == 0:
        return  # real piper_phonemize (or an up-to-date shim) already satisfies this

    sp = _site_packages()
    shim_dir = sp / "piper_phonemize"
    shim_dir.mkdir(exist_ok=True)
    (shim_dir / "__init__.py").write_text(_SHIM_CODE, encoding="utf-8")

    # Fake dist-info so pip treats piper-phonemize 1.1.0 as already installed
    dist_info = sp / "piper_phonemize-1.1.0.dist-info"
    dist_info.mkdir(exist_ok=True)
    (dist_info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: piper-phonemize\nVersion: 1.1.0\n",
        encoding="utf-8",
    )
    (dist_info / "WHEEL").write_text(
        "Wheel-Version: 1.0\nGenerator: faily-shim\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        encoding="utf-8",
    )


# ── audio helpers ─────────────────────────────────────────────────────────────

def _resample_to_22k(src: Path, dest: Path):
    audio, sr = sf.read(str(src), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != _SR:
        audio = sps.resample_poly(audio, _SR, sr).astype(np.float32)
    sf.write(str(dest), audio, _SR)


def _prep_dataset(clips: list[dict], dataset_dir: Path) -> int:
    wavs_dir = dataset_dir / "wavs"
    wavs_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, entry in enumerate(clips):
        transcript = entry.get("transcript", "").strip()
        if not transcript:
            continue
        stem = f"clip_{i:04d}"
        _resample_to_22k(entry["audio"], wavs_dir / f"{stem}.wav")
        rows.append((stem, transcript))
    with open(dataset_dir / "metadata.csv", "w", newline="", encoding="utf-8") as f:
        csv.writer(f, delimiter="|").writerows(rows)
    return len(rows)


async def _stream(cmd: list[str], log_cb, proc_ref: list):
    log_cb(f"$ {' '.join(cmd)}")
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    proc_ref[0] = proc
    log_cb(f"[pid {proc.pid}] started")
    tail: list[str] = []
    n_lines = 0
    async for raw in proc.stdout:
        line = raw.decode(errors="replace").rstrip()
        log_cb(line)
        n_lines += 1
        tail.append(line)
        if len(tail) > 40:
            tail.pop(0)
    await proc.wait()
    proc_ref[0] = None
    log_cb(f"[pid {proc.pid}] exited {proc.returncode} ({n_lines} output lines)")
    if proc.returncode not in (0, -15):
        detail = "\n".join(tail).strip()
        step = cmd[2] if len(cmd) > 2 and cmd[1] == "-m" else cmd[0]
        raise RuntimeError(
            f"{step} exited {proc.returncode}" + (f":\n{detail}" if detail else "")
        )


async def _probe(py: str, label: str, code: str, log_cb) -> None:
    """Run a quick `python -c <code>` in the piper venv and log the outcome.
    Used for pre-flight checks so import-chain breaks show up immediately
    with a clear label, instead of surfacing later inside a much longer
    preprocess/train subprocess log.

    The blocking subprocess.run() runs in a worker thread — Faily (like most
    asyncio UI apps) runs everything on a single event loop, so a blocking
    call here would freeze the whole server, including the websocket pings
    that keep the browser connected, for its full duration (up to the 120s
    timeout). log_cb runs back on the main thread after the await, since
    NiceGUI elements aren't safe to touch from a worker thread.

    These are best-effort diagnostics, not a gate — a cold `import torch` can
    legitimately take well over 30s on Windows (cu128's DLLs are large and
    first-import disk/AV-scan cost is real), so a timeout here is logged and
    swallowed rather than raised. Failing the probe would abort the whole
    training run over a diagnostic that never blocks the actual training
    subprocess later, which has no timeout at all.
    """
    try:
        r = await asyncio.to_thread(
            subprocess.run, [py, "-c", code], capture_output=True, timeout=120, text=True
        )
    except subprocess.TimeoutExpired:
        log_cb(f"[probe:{label}] TIMED OUT after 120s (continuing — this is just a diagnostic)")
        return
    if r.returncode == 0:
        out = r.stdout.strip()
        log_cb(f"[probe:{label}] OK" + (f" — {out}" if out else ""))
    else:
        detail = (r.stderr or r.stdout or "").strip().splitlines()
        log_cb(f"[probe:{label}] FAILED — {detail[-1] if detail else f'exit {r.returncode}'}")


# ── public API ────────────────────────────────────────────────────────────────

async def train(
    clips: list[dict],
    char_dir: Path,
    log_cb,
    proc_ref: list,
    max_epochs: int = 1000,
    base_voice: str = "lessac",
) -> list[tuple[int, Path]]:
    """Train piper voice model. Streams log lines via log_cb.

    Every saved checkpoint gets exported to its own ONNX file for preview —
    returns a list of (epoch, onnx_path) pairs, newest checkpoint last. Call
    finalize_piper_model() with the caller's pick to promote it and discard
    the rest.
    """
    log_cb(f"Character dir: {char_dir}")
    log_cb(f"Clips received: {len(clips)}")

    reason = await asyncio.to_thread(diagnose)
    if reason:
        raise RuntimeError(f"Piper not ready — {reason}")
    log_cb("Piper venv OK")

    usable = [c for c in clips if c.get("transcript", "").strip()]
    log_cb(f"Usable clips (with transcript): {len(usable)}/{len(clips)}")
    if len(usable) < 2:
        raise ValueError(
            f"Need ≥2 clips with transcripts (have {len(usable)}/{len(clips)}). "
            "Add transcripts via the EDIT button on each ref clip."
        )

    if not base_voice_ready(base_voice):
        raise RuntimeError(
            f"Base voice '{base_voice}' isn't downloaded — pick it again to download it first."
        )
    base_ckpt, base_cfg = base_voice_paths(base_voice)
    voice_label = BASE_VOICES.get(base_voice, {}).get("label", base_voice)
    log_cb(f"Base voice: {voice_label}")
    log_cb(f"Base checkpoint: {base_ckpt.name}")
    log_cb(f"Base config: {base_cfg.name}")

    await asyncio.to_thread(_ensure_phonemize_shim)
    log_cb("piper_phonemize shim ready")

    dataset_dir = char_dir / "piper_dataset"
    train_dir   = char_dir / "piper_train"
    train_dir.mkdir(parents=True, exist_ok=True)
    log_cb(f"Dataset dir: {dataset_dir}")
    log_cb(f"Train dir: {train_dir}")

    py = str(_python())
    log_cb(f"Python: {py}")
    log_cb("Running pre-flight import checks…")
    await _probe(py, "torch", "import torch; print(torch.__version__)", log_cb)
    await _probe(
        py, "cuda",
        "import torch; print('available' if torch.cuda.is_available() else 'NOT AVAILABLE — training will fail or fall back to CPU')",
        log_cb,
    )
    await _probe(py, "pytorch_lightning", "import pytorch_lightning; print(pytorch_lightning.__version__)", log_cb)
    await _probe(py, "piper_train.vits.lightning", "import piper_train.vits.lightning", log_cb)

    n = await asyncio.to_thread(_prep_dataset, usable, dataset_dir)
    log_cb(f"Dataset ready: {n} clips")

    # preprocess.py computes batch_size = num_utterances // (max_workers * 2)
    # and errors out ("n must be at least one") if that hits 0 — which it does
    # for small datasets (a handful of ref clips) on any machine with more than
    # a few CPU cores, since it defaults max_workers to os.cpu_count(). Cap it
    # so batch_size always comes out >= 1.
    max_workers = max(1, min(os.cpu_count() or 4, n // 2))
    log_cb(f"Preprocessing (phonemization)… max-workers={max_workers}")
    await _stream([
        py, "-m", "piper_train.preprocess",
        "--language", "en-us",
        "--input-dir",  str(dataset_dir),
        "--output-dir", str(train_dir),
        "--sample-rate", str(_SR),
        "--dataset-format", "ljspeech",
        "--single-speaker",
        "--max-workers", str(max_workers),
    ], log_cb, proc_ref)
    log_cb("Preprocessing done")

    # The base checkpoint already has a current_epoch baked in from its own
    # original training run. --max_epochs is an ABSOLUTE target, not a count
    # of additional epochs — passing our max_epochs directly would set a
    # target already in the past and Lightning refuses to resume
    # ("You restored a checkpoint with current_epoch=2164, but you have set
    # Trainer(max_epochs=1000)"). Treat our max_epochs as "how many more
    # epochs to fine-tune for" and add it on top of the checkpoint's epoch.
    # Read the epoch from inside the checkpoint itself, not its filename —
    # not every base voice's filename follows the "epoch=N-..." convention.
    log_cb("Reading base checkpoint epoch…")
    base_epoch = await _read_checkpoint_epoch(py, base_ckpt)
    target_epochs = base_epoch + max_epochs
    log_cb(
        f"Training — {max_epochs} additional epochs (base checkpoint at epoch "
        f"{base_epoch}, target {target_epochs}), batch 16, GPU, resuming from "
        f"{base_ckpt.name}…"
    )
    await _stream([
        py, "-m", "piper_train",
        "--dataset-dir", str(train_dir),
        "--accelerator", "gpu",
        "--devices", "1",
        "--batch-size", "16",
        "--validation-split", "0.0",
        "--num-test-examples", "0",
        "--max_epochs", str(target_epochs),
        "--resume_from_checkpoint", str(base_ckpt),
        "--checkpoint-epochs", "100",
        "--precision", "32",
        "--default_root_dir", str(train_dir),
    ], log_cb, proc_ref)
    log_cb("Training done")

    ckpts = sorted(train_dir.rglob("*.ckpt"), key=lambda p: p.stat().st_mtime)
    log_cb(f"Checkpoints found: {len(ckpts)}")
    if not ckpts:
        raise RuntimeError("Training finished but no checkpoint was written")

    # Each saved checkpoint is a complete, independent snapshot of the model
    # at that epoch — not a merge of earlier ones. With a tiny dataset and no
    # validation split, the LAST checkpoint isn't necessarily the best-sounding
    # one (it can overfit past a point that sounded more natural). Export every
    # saved checkpoint to its own ONNX file so the caller can preview and pick.
    # Capped as a safety net — --checkpoint-epochs 100 means a normal run only
    # produces a handful; this just guards against pathologically long runs.
    ckpts = ckpts[-20:]

    candidates_dir = char_dir / "piper_candidates"
    if candidates_dir.exists():
        shutil.rmtree(candidates_dir)
    candidates_dir.mkdir(parents=True)

    candidates: list[tuple[int, Path]] = []
    for ckpt in ckpts:
        m = re.search(r"epoch=(\d+)", ckpt.name)
        epoch = int(m.group(1)) if m else 0
        onnx_path = candidates_dir / f"epoch_{epoch:05d}.onnx"
        log_cb(f"Exporting checkpoint (epoch {epoch}) to ONNX…")
        await _stream([
            py, "-m", "piper_train.export_onnx",
            str(ckpt), str(onnx_path),
        ], log_cb, proc_ref)
        shutil.copy2(str(base_cfg), str(onnx_path.with_suffix(".onnx.json")))
        candidates.append((epoch, onnx_path))

    log_cb(f"✓  Exported {len(candidates)} checkpoint candidate(s) for preview")
    return candidates


def finalize_piper_model(char_dir: Path, chosen_onnx: Path) -> Path:
    """Promote a chosen checkpoint candidate (from train()'s return value) to
    the character's permanent Piper model, and discard the other exported
    candidates. Blocking file I/O — small files, safe to call directly."""
    final_onnx = char_dir / "piper.onnx"
    final_cfg = char_dir / "piper.onnx.json"
    shutil.copy2(str(chosen_onnx), str(final_onnx))
    shutil.copy2(str(chosen_onnx.with_suffix(".onnx.json")), str(final_cfg))
    candidates_dir = chosen_onnx.parent
    if candidates_dir.name == "piper_candidates" and candidates_dir.is_dir():
        shutil.rmtree(candidates_dir, ignore_errors=True)
    return final_onnx


def infer(
    text: str,
    model_path: Path,
    out_path: Path,
    length_scale: float = 1.0,
    noise_scale: float = 0.667,
    noise_w_scale: float = 0.8,
) -> Path:
    """Blocking piper inference. Returns out_path.

    length_scale: phoneme length — lower is faster/higher-pitched speech, higher is slower.
    noise_scale: generator noise — controls prosodic variability/expressiveness.
    noise_w_scale: phoneme duration noise — controls pacing variability between phonemes.
    Defaults match piper's own (and the values export_onnx.py bakes into the dummy
    input used to export the model in the first place).
    """
    cfg = model_path.with_suffix(".onnx.json")
    if not cfg.exists():
        raise FileNotFoundError(f"Piper config missing: {cfg}")
    piper = _piper_bin()
    if not piper.exists():
        raise FileNotFoundError(f"Piper binary missing: {piper}")
    # --cuda is safe to pass even without onnxruntime-gpu installed: onnxruntime
    # just warns ("Specified provider 'CUDAExecutionProvider' is not in
    # available provider names") and falls back to CPU. Actually getting GPU
    # acceleration additionally requires onnxruntime-gpu in piper_venv.
    result = subprocess.run(
        [
            str(piper), "--model", str(model_path), "--config", str(cfg),
            "--output_file", str(out_path), "--cuda",
            "--length-scale", str(length_scale),
            "--noise-scale", str(noise_scale),
            "--noise-w-scale", str(noise_w_scale),
        ],
        input=text.encode(),
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Piper: {result.stderr.decode()}")
    return out_path
