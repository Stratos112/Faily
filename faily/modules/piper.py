"""Piper TTS training and inference via isolated subprocess."""
import asyncio
import csv
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


def _python() -> Path:
    return PIPER_VENV / ("Scripts/python.exe" if _IS_WIN else "bin/python")


def _piper_bin() -> Path:
    return PIPER_VENV / ("Scripts/piper.exe" if _IS_WIN else "bin/piper")


def can_infer() -> bool:
    return _piper_bin().exists()


def can_train() -> bool:
    if _IS_WIN:
        return False
    try:
        import subprocess
        r = subprocess.run(
            [str(_python()), "-c", "import piper_train"],
            capture_output=True, timeout=10,
        )
        return r.returncode == 0
    except Exception:
        return False


def is_ready() -> bool:
    return can_infer()


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
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    proc_ref[0] = proc
    async for raw in proc.stdout:
        log_cb(raw.decode(errors="replace").rstrip())
    await proc.wait()
    proc_ref[0] = None
    if proc.returncode not in (0, -15, 1):
        raise RuntimeError(f"Process exited {proc.returncode}")


async def train(
    clips: list[dict],
    char_dir: Path,
    log_cb,
    proc_ref: list,
    max_epochs: int = 1000,
) -> Path:
    """Train piper voice model. Streams log lines via log_cb. Returns .onnx path."""
    if not can_train():
        raise RuntimeError(
            "Piper train not set up — run scripts/setup_piper.bat (Windows) or scripts/setup_piper.sh (WSL2)"
        )

    usable = [c for c in clips if c.get("transcript", "").strip()]
    if len(usable) < 2:
        raise ValueError(
            f"Need ≥2 clips with transcripts (have {len(usable)}/{len(clips)}). "
            "Add transcripts via the EDIT button on each ref clip."
        )

    base_ckpts = sorted(PIPER_CKPTS.glob("*.ckpt")) if PIPER_CKPTS.exists() else []
    base_cfgs  = sorted(PIPER_CKPTS.glob("*.json")) if PIPER_CKPTS.exists() else []
    if not base_ckpts:
        raise RuntimeError("No base checkpoint in piper_checkpoints/ — run setup_piper script first")
    if not base_cfgs:
        raise RuntimeError("No .onnx.json config in piper_checkpoints/ — run setup_piper script first")

    dataset_dir = char_dir / "piper_dataset"
    train_dir   = char_dir / "piper_train"
    train_dir.mkdir(parents=True, exist_ok=True)

    py = str(_python())

    n = _prep_dataset(usable, dataset_dir)
    log_cb(f"Dataset ready: {n} clips")

    log_cb("Preprocessing (phonemization)…")
    await _stream([
        py, "-m", "piper_train.preprocess",
        "--language", "en-us",
        "--input-dir",  str(dataset_dir),
        "--output-dir", str(train_dir),
        "--dataset-format", "ljspeech",
        "--single-speaker",
    ], log_cb, proc_ref)

    log_cb(f"Training — {max_epochs} epochs, batch 16, GPU…")
    await _stream([
        py, "-m", "piper_train",
        "--dataset-dir", str(train_dir),
        "--accelerator", "gpu",
        "--devices", "1",
        "--batch-size", "16",
        "--validation-split", "0.0",
        "--num-test-examples", "0",
        "--max_epochs", str(max_epochs),
        "--resume_from_checkpoint", str(base_ckpts[-1]),
        "--checkpoint-epochs", "100",
        "--precision", "32",
        "--default_root_dir", str(train_dir),
    ], log_cb, proc_ref)

    ckpts = sorted(train_dir.rglob("*.ckpt"), key=lambda p: p.stat().st_mtime)
    if not ckpts:
        raise RuntimeError("Training finished but no checkpoint was written")

    onnx_path = char_dir / "piper.onnx"
    log_cb("Exporting to ONNX…")
    await _stream([
        py, "-m", "piper_train.export_onnx",
        str(ckpts[-1]), str(onnx_path),
    ], log_cb, proc_ref)

    shutil.copy2(str(base_cfgs[-1]), str(char_dir / "piper.onnx.json"))
    log_cb(f"✓  {onnx_path.name}")
    return onnx_path


def infer(text: str, model_path: Path, out_path: Path) -> Path:
    """Blocking piper inference. Returns out_path."""
    cfg = model_path.with_suffix(".onnx.json")
    if not cfg.exists():
        raise FileNotFoundError(f"Piper config missing: {cfg}")
    piper = _piper_bin()
    if not piper.exists():
        raise FileNotFoundError(f"Piper binary missing: {piper}")
    result = subprocess.run(
        [str(piper), "--model", str(model_path), "--config", str(cfg), "--output_file", str(out_path)],
        input=text.encode(),
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Piper: {result.stderr.decode()}")
    return out_path
