# Faily — single image, two venvs (mirrors scripts/setup.bat + scripts/setup_piper.bat)
#
# Main app: Python 3.12 (not 3.14 — pyproject only requires >=3.10, and 3.12 has
# reliably-published Linux wheels for the entire ML dependency list; 3.14 was
# just what ended up on the Windows box this was ported from).
# Piper training: Python 3.11, piper-train's own pin — unrelated to the above.
#
# Data downloads that land in volume-mounted directories (models/, piper_checkpoints/)
# are intentionally NOT done here — anything baked into those paths at build time
# would just be shadowed by the bind mount at container start. See entrypoint.sh.

FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# ── system packages ───────────────────────────────────────────────────────────
# python3.11 isn't reliably in Ubuntu 24.04's own repos (24.04 defaults to 3.12);
# deadsnakes provides it. espeak-ng is piper-train's phonemization backend.
# libportaudio2/libsndfile1 are runtime shared libs some audio deps dlopen.
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common gnupg ca-certificates curl git \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3.12-dev \
        python3.11 python3.11-venv python3.11-dev \
        build-essential \
        espeak-ng libportaudio2 libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# ── main app venv (Python 3.12) ───────────────────────────────────────────────
RUN python3.12 -m venv /app/.venv
ENV PATH="/app/.venv/bin:${PATH}"

# PyTorch cu128 first (large, changes rarely — own layer)
RUN pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cu128

# Core Faily deps, listed explicitly (mirrors pyproject.toml's [project.dependencies])
# rather than `pip install -e .`, so this layer doesn't depend on the app source
# being present yet — app code is copied later, near the end of the build.
RUN pip install --no-cache-dir \
        "nicegui>=2.0.0" \
        "transformers>=4.40.0" \
        "diffusers>=0.27.0" \
        "accelerate>=0.27.0" \
        "soundfile>=0.12.0" \
        "numpy>=1.24.0" \
        "huggingface_hub>=0.22.0" \
        "scipy>=1.11.0" \
        "speechbrain>=1.0.0" \
        "noisereduce>=3.0.0"

# Voice cloning backends (CLONE / ONE SHOT / SPEAK stage-2)
RUN pip install --no-cache-dir coqui-tts \
    ; pip install --no-cache-dir f5-tts \
    ; pip install --no-cache-dir chatterbox-tts

# Expression engine (SPEAK stage-1)
RUN pip install --no-cache-dir parler-tts

# OpenVoice v2 (SPEAK stage-2 voice conversion) — --no-deps avoids pinned
# version conflicts (gradio 3.x, numpy 1.22, librosa 0.9, etc.)
RUN pip install --no-cache-dir --no-deps "git+https://github.com/myshell-ai/OpenVoice.git" \
    && pip install --no-cache-dir wavmark resampy faster-whisper cn2an eng_to_ipa langid jieba

# Seed-VC (SPEAK stage-2 zero-shot voice conversion) runtime deps — the repo
# itself is cloned at container startup into the volume-mounted models/ dir
# (see entrypoint.sh), not here.
RUN pip install --no-cache-dir \
        bigvgan munch einops descript-audio-codec resemblyzer pydub \
        hydra-core python-dotenv sounddevice jiwer

# App source (copied late so the above layers stay cached across code changes)
COPY faily/ /app/faily/
COPY main.py pyproject.toml README.md /app/
COPY scripts/ /app/scripts/
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Register the faily package itself (deps already installed above, so --no-deps
# keeps this fast and avoids re-resolving anything from PyPI).
RUN pip install --no-cache-dir --no-deps -e /app

# bigvgan._from_pretrained / huggingface_hub hub_mixin one-time patches
RUN python /app/scripts/patch_bigvgan.py \
    ; python /app/scripts/patch_hub_mixin.py

# ── piper training venv (Python 3.11) ─────────────────────────────────────────
RUN python3.11 -m venv /app/piper_venv
RUN /app/piper_venv/bin/python -m pip install "pip<24.1" --quiet \
    && /app/piper_venv/bin/python -m pip uninstall pytorch-lightning -y 2>/dev/null || true

RUN /app/piper_venv/bin/pip install --no-cache-dir piper-tts phonemizer \
    && /app/piper_venv/bin/pip install --no-cache-dir cython librosa six \
    && /app/piper_venv/bin/pip install --no-cache-dir "torchmetrics==0.11.4" "pytorch-lightning~=1.7.0"

# pytorch-lightning 1.7.x source patches for modern NumPy/PyTorch (np.Inf alias,
# torch.load weights_only default, cross-platform checkpoint loading)
RUN /app/piper_venv/bin/python /app/scripts/patch_pytorch_lightning.py

RUN /app/piper_venv/bin/pip install --no-cache-dir --no-deps \
        "piper-train @ git+https://github.com/rhasspy/piper.git#subdirectory=src/python"

# VAD model — piper-train's setup.py doesn't declare it as package data, so
# pip's wheel build silently drops it (lands inside the venv's site-packages,
# which is image-baked, not volume-mounted — safe to fetch at build time).
RUN VAD_DIR=/app/piper_venv/lib/python3.11/site-packages/piper_train/norm_audio/models \
    && mkdir -p "$VAD_DIR" \
    && curl -L -o "$VAD_DIR/silero_vad.onnx" \
        "https://raw.githubusercontent.com/rhasspy/piper/master/src/python/piper_train/norm_audio/models/silero_vad.onnx"

# monotonic_align — same package-data gap drops core.pyx entirely; fetch and
# build it ourselves. Deliberately the STOCK extension here: the whole
# serial-loop/noexcept/nan-check/bounds-check/pure-python patch trail in
# scripts/patch_piper_train_monotonic_align_*.py was chasing an MSVC/Windows-
# specific native crash. gcc's OpenMP is a different, more mature code path —
# start clean on Linux and only apply the pure-Python fallback
# (patch_piper_train_monotonic_align_pure_python.py) if it turns out to
# reproduce here too.
RUN MONO_DIR=/app/piper_venv/lib/python3.11/site-packages/piper_train/vits/monotonic_align \
    && curl -L -o "$MONO_DIR/core.pyx" \
        "https://raw.githubusercontent.com/rhasspy/piper/master/src/python/piper_train/vits/monotonic_align/core.pyx" \
    && cd /app/piper_venv/lib/python3.11/site-packages \
    && /app/piper_venv/bin/python piper_train/vits/monotonic_align/setup.py build_ext --inplace \
    && python3 -c "from pathlib import Path; p = Path('$MONO_DIR/__init__.py'); t = p.read_text(); p.write_text(t.replace('from .monotonic_align.core import', 'from .core import'))"

# export_onnx.py needs the base `onnx` package (piper-train's requirements.txt
# only declares onnxruntime, which is inference-only)
RUN /app/piper_venv/bin/pip install --no-cache-dir onnx

# torch.onnx.export dynamo-exporter default (PyTorch 2.9+) / DataLoader
# worker-subprocess patches
RUN /app/piper_venv/bin/python /app/scripts/patch_piper_train_onnx.py \
    && /app/piper_venv/bin/python /app/scripts/patch_piper_train_num_workers.py

# Replace torch with cu128 for Blackwell GPUs (piper-train pulls 1.13.1+cu117,
# which can't run on this hardware at all)
RUN /app/piper_venv/bin/pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu128 --force-reinstall

EXPOSE 7842

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["/app/.venv/bin/python", "main.py"]
