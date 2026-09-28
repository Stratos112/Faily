#!/usr/bin/env bash
# setup_linux.sh — the one install path for Faily on Linux.
#
# Used both by the Dockerfile (one phase per layer, so layers cache) and
# directly in a dev box / bare server (no args = every phase). Every phase is
# idempotent — rerunning skips what's already done.
#
#   bash scripts/setup_linux.sh                 # system main piper app data
#   bash scripts/setup_linux.sh main piper      # just those phases
#
# Phases:
#   system  apt packages (espeak-ng, ffmpeg, build tools, audio libs) + uv
#   main    .venv (Python 3.14): torch cu128, core deps, cloning/expression backends
#   piper   piper_venv (Python 3.11): piper-tts + piper-train + patches, torch cu128
#   app     register the faily package itself into .venv (needs the source tree)
#   data    Seed-VC repo + Piper base checkpoint (writes into volume-mounted dirs,
#           so the container runs this at startup — see entrypoint.sh)
#
# Why Python 3.14 for the main venv: chatterbox-tts pins torch==2.6.0 on
# Python < 3.14 (torch>=2.9 only on 3.14+), and torch 2.6 has no Blackwell
# (sm_120) kernels. 3.14 is also what the known-working native-Windows setup
# ran. Interpreters come from uv (python-build-standalone) so this works the
# same on Debian 12 and Ubuntu 24.04 without deadsnakes; they live in
# $PROJECT_DIR/.python so venvs don't dangle when a dev container is rebuilt.
set -euo pipefail

PROJECT_DIR="${FAILY_PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
SCRIPT_DIR="$PROJECT_DIR/scripts"
VENV="$PROJECT_DIR/.venv"
PIPER_VENV="$PROJECT_DIR/piper_venv"
TORCH_INDEX="https://download.pytorch.org/whl/cu128"
MAIN_PY_VERSION="3.14"
PIPER_PY_VERSION="3.11"   # piper-train's own pin
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$PROJECT_DIR/.python}"
export PIP_DISABLE_PIP_VERSION_CHECK=1

WARNINGS=()
say()  { echo; echo "── $* ──"; }
warn() { echo "WARNING: $*" >&2; WARNINGS+=("$*"); }

as_root() {
    if [ "$(id -u)" -eq 0 ]; then "$@"; else sudo "$@"; fi
}

uv_bin() {
    command -v uv 2>/dev/null || { [ -x "$HOME/.local/bin/uv" ] && echo "$HOME/.local/bin/uv"; } || true
}

# Path to a uv-managed interpreter of the given version (installed on demand).
managed_python() {
    local uv; uv="$(uv_bin)"
    [ -n "$uv" ] || { echo "ERROR: uv not found — run the 'system' phase first." >&2; exit 1; }
    # --no-bin: don't drop python3.x shims into ~/.local/bin — only our venvs use these.
    "$uv" python install --no-bin "$1" >&2
    "$uv" python find --managed-python "$1"
}

# Install packages, but don't abort the whole setup if they fail — mirrors
# setup.bat's "WARNING: X failed, feature Y won't work" behavior. On failure,
# retry one by one so a single unbuildable package doesn't take its
# neighbours down with it, and report exactly which ones are missing.
pip_soft() {
    local pip="$1" label="$2"; shift 2
    if "$pip" install "$@"; then return 0; fi
    local flags=() pkgs=() a
    for a in "$@"; do
        if [[ "$a" == -* ]]; then flags+=("$a"); else pkgs+=("$a"); fi
    done
    [ ${#pkgs[@]} -gt 1 ] || { warn "$label: install failed (${pkgs[*]})"; return 0; }
    for a in "${pkgs[@]}"; do
        "$pip" install "${flags[@]}" "$a" || warn "$label: $a failed to install"
    done
}

# Prints "ok" if the venv's torch can run on Blackwell (CUDA build >= 12.8)
# and torchaudio (if installed) matches torch's version; otherwise the reason.
torch_status() {
    "$1/bin/python" - <<'PY' 2>/dev/null || echo "torch not importable"
import importlib.util
import torch
cuda = torch.version.cuda
if not cuda or tuple(map(int, cuda.split(".")[:2])) < (12, 8):
    print(f"torch {torch.__version__} has CUDA {cuda} (need >= 12.8 for sm_120)")
elif importlib.util.find_spec("torchaudio"):
    try:
        import torchaudio
    except Exception as e:
        print(f"torchaudio fails to import against torch {torch.__version__}: {type(e).__name__}")
        raise SystemExit
    mm = lambda v: v.split("+")[0].split(".")[:2]
    print("ok" if mm(torch.__version__) == mm(torchaudio.__version__)
          else f"torch {torch.__version__} / torchaudio {torchaudio.__version__} mismatch")
else:
    print("ok")
PY
}

# Force torch(/torchaudio) onto a Blackwell-capable cu128 build. Run LAST in
# each venv: any later install whose resolver decides it wants a different
# torch would otherwise silently swap in a build with no sm_120 kernels.
ensure_cu128_torch() {
    local venv="$1" name status; shift
    name="$(basename "$venv")"
    status="$(torch_status "$venv")"
    if [ "$status" = "ok" ]; then
        echo "torch OK in $name: $("$venv/bin/python" -c 'import torch; print(torch.__version__)')"
        return 0
    fi
    echo "$name: $status — installing torch cu128..."
    "$venv/bin/pip" install --upgrade "$@" --index-url "$TORCH_INDEX"
    status="$(torch_status "$venv")"
    # --upgrade treats e.g. 2.12.1+cu130 and 2.12.1+cu128 as the same version
    # and leaves a mismatched pair alone — reinstall just these packages cleanly.
    if [ "$status" != "ok" ]; then
        echo "$name: still $status — clean reinstall of $*"
        "$venv/bin/pip" uninstall -y "$@"
        "$venv/bin/pip" install "$@" --index-url "$TORCH_INDEX"
        status="$(torch_status "$venv")"
    fi
    [ "$status" = "ok" ] || { echo "ERROR: $name: $status" >&2; exit 1; }
}

# Create a venv on the given Python version. An existing venv on a different
# version (e.g. an old system-python dev venv) is moved aside, not reused —
# the version decides which torch the backends' pins resolve to.
make_venv() {
    local venv="$1" want="$2" have
    if [ -x "$venv/bin/python" ]; then
        have="$("$venv/bin/python" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || echo broken)"
        [ "$have" = "$want" ] && return 0
        local aside="$venv.old-py${have//./}"
        warn "$(basename "$venv") was Python $have (want $want) — moved to $(basename "$aside")"
        rm -rf "$aside"
        mv "$venv" "$aside"
    fi
    "$(managed_python "$want")" -m venv "$venv"
}

# ══════════════════════════════════════════════════════════════════════════════
phase_system() {
    say "system packages"
    as_root apt-get update
    as_root env DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ca-certificates curl git build-essential \
        espeak-ng ffmpeg libsndfile1 libportaudio2
    # Keep the image slim; harmless elsewhere (every run starts with apt-get update).
    as_root rm -rf /var/lib/apt/lists/*

    say "uv"
    if [ -n "$(uv_bin)" ]; then
        echo "uv already installed: $(uv_bin)"
    else
        # Honors UV_INSTALL_DIR (the Dockerfile sets /usr/local/bin); defaults to ~/.local/bin.
        curl -LsSf https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh
    fi
}

# ══════════════════════════════════════════════════════════════════════════════
phase_main() {
    say "main venv (Python $MAIN_PY_VERSION)"
    make_venv "$VENV" "$MAIN_PY_VERSION"
    local pip="$VENV/bin/pip"
    "$pip" install --upgrade pip

    say "torch cu128"
    ensure_cu128_torch "$VENV" torch torchaudio

    say "core deps (from pyproject.toml)"
    # Read [project.dependencies] rather than `pip install -e .` so this works
    # before the app source is copied into the image (keeps this layer cached
    # across code changes). tomllib is stdlib on 3.11+.
    local core
    mapfile -t core < <("$VENV/bin/python" -c "
import tomllib
print('\n'.join(tomllib.load(open('$PROJECT_DIR/pyproject.toml', 'rb'))['project']['dependencies']))
")
    "$pip" install "${core[@]}"

    say "voice cloning backends"
    pip_soft "$pip" "coqui-tts (XTTS v2 + FreeVC)" coqui-tts
    pip_soft "$pip" "f5-tts"                      f5-tts
    pip_soft "$pip" "chatterbox-tts"              chatterbox-tts

    say "expression engine: parler-tts"
    # --no-deps: parler-tts pins transformers==4.46.1 exactly, which would
    # downgrade the transformers 5.x that chatterbox pins and that vc.py's
    # _load_parler compatibility patches are written against.
    pip_soft "$pip" "parler-tts" --no-deps parler-tts
    pip_soft "$pip" "parler-tts deps" \
        sentencepiece descript-audio-codec-unofficial descript-audiotools-unofficial "protobuf>=4.0.0"

    say "OpenVoice v2"
    # --no-deps avoids its stale pins (gradio 3.x, numpy 1.22, librosa 0.9...).
    pip_soft "$pip" "OpenVoice" --no-deps "git+https://github.com/myshell-ai/OpenVoice.git"
    pip_soft "$pip" "OpenVoice deps" wavmark resampy faster-whisper cn2an eng_to_ipa langid jieba

    say "Seed-VC runtime deps"
    # Selective — the repo's requirements.txt pins torch==2.4 / numpy==1.26.4 /
    # transformers==4.46.3. The repo itself is cloned by the 'data' phase.
    pip_soft "$pip" "Seed-VC deps" \
        bigvgan munch einops descript-audio-codec resemblyzer pydub \
        hydra-core python-dotenv sounddevice jiwer

    say "repair shared deps clobbered by backend pins"
    # Each backend install above re-resolves shared packages to satisfy its
    # own (often stale) pins, and pip's resolver doesn't protect what's
    # already installed. Two of those actually break things at runtime:
    #  - chatterbox pins gradio==6.8.0 → starlette<1.0, under nicegui (needs
    #    >=1.3.1) — the whole UI. Reinstalling Faily's own deps fixes that.
    #  - descript-audiotools (Seed-VC/parler) pins protobuf<3.20, but onnx
    #    (chatterbox's s3tokenizer), wandb (f5-tts) and onnxruntime need >=5/6.
    #    audiotools never imports protobuf itself — the pin is stale.
    # Leftover `pip check` complaints after this are unused code paths only
    # (gradio demos, audiotools' pins, parler's transformers pin).
    "$pip" install "${core[@]}"
    "$pip" install "protobuf>=6.31.1,<8"

    say "torch cu128 (re-check after backend installs)"
    ensure_cu128_torch "$VENV" torch torchaudio

    say "pip check (main venv) — conflicts here are expected, see comments above"
    "$pip" check || true
}

# ══════════════════════════════════════════════════════════════════════════════
phase_piper() {
    say "piper venv (Python $PIPER_PY_VERSION)"
    if [ -d "$PIPER_VENV/Scripts" ] && [ ! -x "$PIPER_VENV/bin/python" ]; then
        echo "ERROR: $PIPER_VENV is a Windows-created venv — delete it and rerun." >&2
        exit 1
    fi
    make_venv "$PIPER_VENV" "$PIPER_PY_VERSION"
    local py="$PIPER_VENV/bin/python" pip="$PIPER_VENV/bin/pip"

    if [ -x "$PIPER_VENV/bin/piper" ] && "$py" -c "import piper_train.vits.monotonic_align, onnx" 2>/dev/null; then
        echo "piper venv already complete — skipping package installs."
    else
        # pip >= 24.1 rejects pytorch-lightning 1.7.x's invalid metadata.
        "$py" -m pip install "pip<24.1"
        "$pip" uninstall -y pytorch-lightning 2>/dev/null || true

        "$pip" install piper-tts phonemizer
        "$pip" install cython librosa six
        # torchmetrics pinned: newer releases dropped a private helper 1.7.x imports.
        "$pip" install "torchmetrics==0.11.4" "pytorch-lightning~=1.7.0"

        # np.Inf alias, torch.load weights_only default, checkpoint loading.
        "$py" "$SCRIPT_DIR/patch_pytorch_lightning.py"

        "$pip" install --no-deps \
            "piper-train @ git+https://github.com/rhasspy/piper.git#subdirectory=src/python"

        local sp; sp="$("$py" -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")"

        # piper-train's setup.py doesn't declare these as package data, so the
        # wheel build silently drops them.
        local vad_dir="$sp/piper_train/norm_audio/models"
        mkdir -p "$vad_dir"
        [ -f "$vad_dir/silero_vad.onnx" ] || curl -fL -o "$vad_dir/silero_vad.onnx" \
            "https://raw.githubusercontent.com/rhasspy/piper/master/src/python/piper_train/norm_audio/models/silero_vad.onnx"

        # monotonic_align: stock Cython extension. The serial/noexcept/nan/
        # bounds patch trail was chasing an MSVC-only crash; start clean on
        # gcc and fall back to patch_piper_train_monotonic_align_pure_python.py
        # only if training crashes here too. Must build with cwd = site-packages
        # (cythonize derives the dotted module name from the parent packages).
        local mono="$sp/piper_train/vits/monotonic_align"
        [ -f "$mono/core.pyx" ] || curl -fL -o "$mono/core.pyx" \
            "https://raw.githubusercontent.com/rhasspy/piper/master/src/python/piper_train/vits/monotonic_align/core.pyx"
        ( cd "$sp" && "$py" piper_train/vits/monotonic_align/setup.py build_ext --inplace )
        # piper-train bug: stale relative import that doesn't match where the
        # extension actually lands.
        sed -i 's/from \.monotonic_align\.core import/from .core import/' "$mono/__init__.py"

        # export_onnx.py needs onnx (piper-train only declares onnxruntime).
        "$pip" install onnx

        # torch>=2.9 dynamo ONNX exporter default; DataLoader worker subprocess.
        "$py" "$SCRIPT_DIR/patch_piper_train_onnx.py"
        "$py" "$SCRIPT_DIR/patch_piper_train_num_workers.py"
    fi

    # piper-train pulls torch 1.13.1+cu117 — no Blackwell kernels at all.
    say "torch cu128 (piper venv)"
    ensure_cu128_torch "$PIPER_VENV" torch
}

# ══════════════════════════════════════════════════════════════════════════════
phase_app() {
    say "faily package"
    [ -d "$PROJECT_DIR/faily" ] || { echo "ERROR: no faily/ source in $PROJECT_DIR" >&2; exit 1; }
    "$VENV/bin/pip" install --no-deps -e "$PROJECT_DIR"
}

# ══════════════════════════════════════════════════════════════════════════════
download() {  # url dest label — atomic (no half-written file left on failure)
    local url="$1" dest="$2" label="$3"
    if [ -f "$dest" ]; then echo "$label already present."; return 0; fi
    echo "Downloading $label..."
    if curl -fL --progress-bar -o "$dest.part" "$url"; then
        mv "$dest.part" "$dest"
    else
        rm -f "$dest.part"
        warn "$label download failed"
    fi
}

phase_data() {
    say "Seed-VC repo"
    local seedvc="$PROJECT_DIR/models/vc/seed-vc"
    if [ -d "$seedvc" ]; then
        echo "Seed-VC repo already present."
    else
        mkdir -p "$(dirname "$seedvc")"
        git clone --quiet https://github.com/Plachtaa/seed-vc.git "$seedvc" \
            || warn "Seed-VC clone failed — Seed-VC voice conversion will not work"
    fi

    say "Piper base checkpoint"
    local ckpts="$PROJECT_DIR/piper_checkpoints"
    mkdir -p "$ckpts"
    download "https://huggingface.co/datasets/rhasspy/piper-checkpoints/resolve/main/en/en_US/lessac/medium/epoch=2164-step=1355540.ckpt" \
        "$ckpts/epoch=2164-step=1355540.ckpt" "Piper base checkpoint (~800 MB)"
    download "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json" \
        "$ckpts/en_US-lessac-medium.onnx.json" "Piper voice config"
}

# ══════════════════════════════════════════════════════════════════════════════
PHASES=("$@")
[ ${#PHASES[@]} -gt 0 ] || PHASES=(system main piper app data)
for phase in "${PHASES[@]}"; do
    case "$phase" in
        system|main|piper|app|data) "phase_$phase" ;;
        *) echo "Unknown phase: $phase (expected system|main|piper|app|data)" >&2; exit 2 ;;
    esac
done

echo
if [ ${#WARNINGS[@]} -gt 0 ]; then
    echo "══ Finished with ${#WARNINGS[@]} warning(s) ══"
    printf '  - %s\n' "${WARNINGS[@]}"
else
    echo "══ Finished: ${PHASES[*]} ══"
fi
