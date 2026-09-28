#!/usr/bin/env bash
# Runs once per container start, before the app. Handles the two setup steps
# that write into volume-mounted directories (models/, piper_checkpoints/) —
# anything baked into those paths at image-build time would just be shadowed
# by the bind mount, so these have to happen at runtime instead. Both are
# idempotent (skip if already present), matching the original setup.bat /
# setup_piper.bat behavior.
set -euo pipefail

# ── Seed-VC repo (code, not weights — used as a local import, not a package) ──
SEEDVC_DIR="/app/models/vc/seed-vc"
if [ ! -d "$SEEDVC_DIR" ]; then
    echo "Cloning Seed-VC repo..."
    mkdir -p "$(dirname "$SEEDVC_DIR")"
    git clone --quiet https://github.com/Plachtaa/seed-vc.git "$SEEDVC_DIR" \
        || echo "WARNING: Seed-VC clone failed — Seed-VC voice conversion will not work."
else
    echo "Seed-VC repo already present."
fi

# ── Piper base checkpoint + voice config ──────────────────────────────────────
CKPT_DIR="/app/piper_checkpoints"
mkdir -p "$CKPT_DIR"

CKPT="$CKPT_DIR/epoch=2164-step=1355540.ckpt"
if [ ! -f "$CKPT" ]; then
    echo "Downloading Piper base checkpoint (~800 MB)..."
    curl -L --progress-bar -o "$CKPT" \
        "https://huggingface.co/datasets/rhasspy/piper-checkpoints/resolve/main/en/en_US/lessac/medium/epoch=2164-step=1355540.ckpt"
else
    echo "Piper base checkpoint already present."
fi

CFG="$CKPT_DIR/en_US-lessac-medium.onnx.json"
if [ ! -f "$CFG" ]; then
    echo "Downloading Piper voice config..."
    curl -L -o "$CFG" \
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json"
else
    echo "Piper voice config already present."
fi

exec "$@"
