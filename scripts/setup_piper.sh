#!/usr/bin/env bash
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
    echo "ERROR: Python 3.9-3.11 required. Install with: sudo apt install python3.11"
    exit 1
fi
echo "Using $PYBIN"

# ── espeak-ng (phonemizer dependency) ───────────────────────────────────────
if ! command -v espeak-ng &>/dev/null; then
    echo "Installing espeak-ng..."
    sudo apt-get install -y espeak-ng
fi

# ── Create venv ──────────────────────────────────────────────────────────────
if [ -f "$VENV/bin/python" ]; then
    echo "Piper venv already exists — skipping creation."
else
    echo "Creating venv at $VENV"
    "$PYBIN" -m venv "$VENV"
    "$VENV/bin/pip" install --upgrade pip
    "$VENV/bin/pip" install piper-train piper-tts
    echo "Venv ready."
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

echo ""
echo "Piper setup complete."
echo "  Venv:       $VENV"
echo "  Checkpoint: $CKPT"
echo "  Config:     $CFG"
