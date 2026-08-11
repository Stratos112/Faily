#!/usr/bin/env bash
# WSL2 / Linux setup for piper training.
# Inference runs on CPU in WSL2 (no GPU passthrough for cu128 torch).
# For GPU training use the Windows setup: scripts/setup_piper.bat
set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$PROJECT_DIR/piper_venv"
CKPT_DIR="$PROJECT_DIR/piper_checkpoints"

# ── Python version ───────────────────────────────────────────────────────────
PYBIN=""
for v in python3.11 python3.10 python3.9; do
    if command -v "$v" &>/dev/null; then
        PYBIN="$v"
        break
    fi
done
if [ -z "$PYBIN" ]; then
    echo "ERROR: Python 3.9-3.11 required."
    echo "  sudo apt install python3.11 python3.11-venv"
    exit 1
fi
echo "Using $($PYBIN --version)"

# ── espeak-ng ────────────────────────────────────────────────────────────────
if ! command -v espeak-ng &>/dev/null; then
    echo "Installing espeak-ng..."
    sudo apt-get install -y espeak-ng
fi

# ── Venv ─────────────────────────────────────────────────────────────────────
# Guard against running this against a Windows-created venv (e.g. this repo
# mounted at /mnt/c or /mnt/e and setup_piper.bat already run there). A
# Windows venv has Scripts/ instead of bin/ — creating a Linux venv on top of
# it corrupts pyvenv.cfg's `home` for the Windows side ("No Python at
# '/usr/bin\python.exe'" when Windows tries to use it afterward).
if [ -d "$VENV/Scripts" ] && [ ! -f "$VENV/bin/python" ]; then
    echo "ERROR: $VENV looks like a Windows-created venv (has Scripts/, no bin/python)."
    echo "  Running this WSL2 setup here would corrupt it for Windows use."
    echo "  If this is genuinely meant to be a WSL2-only venv, remove $VENV first."
    exit 1
fi

if [ -f "$VENV/bin/piper" ] && "$VENV/bin/python" -c "import piper_train" 2>/dev/null; then
    echo "Piper venv already complete — skipping."
    SKIP_INSTALL=1
fi

if [ -z "$SKIP_INSTALL" ]; then
    if [ ! -f "$VENV/bin/python" ]; then
        echo "Creating venv..."
        "$PYBIN" -m venv "$VENV"
    fi

    # Downgrade pip so pytorch-lightning 1.7.x metadata is accepted
    echo "Downgrading pip to allow pytorch-lightning 1.7.x..."
    "$VENV/bin/pip" install "pip<24.1" --quiet

    echo "Installing piper-tts..."
    "$VENV/bin/pip" install piper-tts --quiet

    # pytorch-lightning 1.7.x (pulled in by piper-train below) declares
    # torchmetrics>=0.7.0 with no upper bound, so pip grabs the newest release,
    # which dropped the private _compare_version helper 1.7.x imports at
    # startup. Pin a version still compatible with 1.7.x before piper-train
    # installs, so its resolver leaves this one alone. Also needed by
    # torch.utils.tensorboard's import chain but not declared anywhere.
    echo "Pinning torchmetrics/six for pytorch-lightning 1.7.x compatibility..."
    "$VENV/bin/pip" install "torchmetrics==0.11.4" six --quiet

    echo "Installing piper-train from source (pulls torch, ~2 GB)..."
    "$VENV/bin/pip" install \
        "piper-train @ git+https://github.com/rhasspy/piper.git#subdirectory=src/python"

    # piper-train's setup.py doesn't declare norm_audio/models/*.onnx as
    # package data, so pip's wheel build silently drops it even though it's a
    # real file in the repo. Without it, preprocessing crashes trying to load
    # the VAD model.
    SITE_PACKAGES=$("$VENV/bin/python" -c "import site; print(site.getsitepackages()[0])")
    VAD_DIR="$SITE_PACKAGES/piper_train/norm_audio/models"
    VAD_FILE="$VAD_DIR/silero_vad.onnx"
    if [ ! -f "$VAD_FILE" ]; then
        echo "Downloading silero_vad.onnx (missing from piper-train package)..."
        mkdir -p "$VAD_DIR"
        curl -L -o "$VAD_FILE" \
          "https://raw.githubusercontent.com/rhasspy/piper/master/src/python/piper_train/norm_audio/models/silero_vad.onnx"
    fi

    # pytorch-lightning 1.7.x needs a couple of source patches to work with a
    # modern NumPy/PyTorch stack (removed np.Inf alias, torch.load's
    # weights_only default flip, cross-platform checkpoint loading). See
    # patch_pytorch_lightning.py for details — inline one-liners here got
    # unwieldy and error-prone.
    echo "Patching pytorch-lightning for NumPy/torch 2.6+ compatibility..."
    "$VENV/bin/python" "$PROJECT_DIR/scripts/patch_pytorch_lightning.py"

    # monotonic_align is a Cython extension piper-train ships as source
    # (.pyx) but never builds — same package-data gap as above drops core.pyx
    # entirely, and its own nested setup.py is never invoked by pip. Fetch
    # the source and build it in place. Must run with cwd = site-packages
    # root: cythonize infers the fully-qualified module name
    # (piper_train.vits.monotonic_align.core) by walking up through parent
    # __init__.py files, and `build_ext --inplace` writes to that path
    # relative to cwd — running it from inside the module's own directory
    # makes it try to create piper_train/vits/monotonic_align/ *inside*
    # itself and fail.
    MONO_DIR="$SITE_PACKAGES/piper_train/vits/monotonic_align"
    if [ ! -f "$MONO_DIR/core.pyx" ]; then
        echo "Downloading monotonic_align Cython source..."
        curl -L -o "$MONO_DIR/core.pyx" \
          "https://raw.githubusercontent.com/rhasspy/piper/master/src/python/piper_train/vits/monotonic_align/core.pyx"
    fi
    echo "Building monotonic_align Cython extension..."
    ( cd "$SITE_PACKAGES" && "$VENV/bin/python" piper_train/vits/monotonic_align/setup.py build_ext --inplace )

    # piper-train's monotonic_align/__init__.py has a stale relative import
    # (from .monotonic_align.core import ...) that doesn't match where the
    # extension we just built actually lands (.core, one level up) — a bug
    # in piper-train itself. Patch it.
    "$VENV/bin/python" -c "
from pathlib import Path
p = Path('$MONO_DIR/__init__.py')
t = p.read_text()
p.write_text(t.replace('from .monotonic_align.core import', 'from .core import'))
"
fi

# ── Download base checkpoint ─────────────────────────────────────────────────
mkdir -p "$CKPT_DIR"

CKPT="$CKPT_DIR/epoch=2164-step=1355540.ckpt"
if [ ! -f "$CKPT" ]; then
    echo "Downloading base checkpoint (~400 MB)..."
    curl -L --progress-bar -o "$CKPT" \
      "https://huggingface.co/datasets/rhasspy/piper-checkpoints/resolve/main/en/en_US/lessac/medium/epoch=2164-step=1355540.ckpt"
else
    echo "Base checkpoint already present."
fi

CFG="$CKPT_DIR/en_US-lessac-medium.onnx.json"
if [ ! -f "$CFG" ]; then
    echo "Downloading voice config..."
    curl -L -o "$CFG" \
      "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json"
else
    echo "Voice config already present."
fi

# ── Diagnostics — installed versions + pip's own conflict check ─────────────
# The pins above (torchmetrics, numpy patches, etc.) exist because pip's
# normal resolver doesn't catch these — `pip check` only flags conflicts
# between DECLARED requirements, not runtime API incompatibilities like
# np.Inf/weights_only. Still useful as a first signal if something's off.
echo ""
echo "── Installed package versions ──────────────────────────────────────────────"
for pkg in torch torchaudio torchmetrics pytorch-lightning numpy cython six piper-tts; do
    "$VENV/bin/pip" show "$pkg" 2>/dev/null | grep -E "^(Name|Version):"
    echo ""
done
echo "── pip check (flags declared-requirement conflicts, if any) ────────────────"
"$VENV/bin/pip" check
echo ""

echo "Verify:"
"$VENV/bin/python" -c "
import piper_train, torch
print(f'piper_train OK | torch {torch.__version__} | CUDA: {torch.cuda.is_available()}')
"
