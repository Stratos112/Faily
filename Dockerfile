# syntax=docker/dockerfile:1
# Faily — single image, two venvs. All install logic lives in
# scripts/setup_linux.sh (shared with bare-metal / dev-box installs); this file
# just runs its phases as separate layers so they cache independently.
#
# Main app: Python 3.14 (chatterbox-tts pins torch 2.6 — no Blackwell kernels —
# on anything older). Piper training: Python 3.11, piper-train's own pin.
# Interpreters come from uv, so no deadsnakes PPA is needed.
#
# Data that lands in volume-mounted directories (models/, piper_checkpoints/)
# is NOT fetched here — it'd be shadowed by the bind mount at container start.
# entrypoint.sh runs the script's `data` phase instead.

FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    UV_INSTALL_DIR=/usr/local/bin \
    UV_PYTHON_INSTALL_DIR=/app/.python \
    PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app

# ── system packages + uv ──────────────────────────────────────────────────────
COPY scripts/setup_linux.sh /app/scripts/
RUN bash scripts/setup_linux.sh system

# ── main app venv ─────────────────────────────────────────────────────────────
# Only the files this phase reads are copied, so code changes don't bust it.
# The pip cache mount keeps wheels across rebuilds (needs BuildKit, which is
# Docker Desktop's default).
COPY pyproject.toml /app/
COPY scripts/patch_bigvgan.py scripts/patch_hub_mixin.py /app/scripts/
RUN --mount=type=cache,target=/root/.cache/pip \
    bash scripts/setup_linux.sh main

# ── piper training venv ───────────────────────────────────────────────────────
COPY scripts/patch_pytorch_lightning.py scripts/patch_piper_train_onnx.py \
     scripts/patch_piper_train_num_workers.py /app/scripts/
RUN --mount=type=cache,target=/root/.cache/pip \
    bash scripts/setup_linux.sh piper

# ── app source (late, so the layers above stay cached across code changes) ────
COPY faily/ /app/faily/
COPY main.py README.md entrypoint.sh /app/
COPY scripts/ /app/scripts/
RUN chmod +x /app/entrypoint.sh && bash scripts/setup_linux.sh app

EXPOSE 7842

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["/app/.venv/bin/python", "main.py"]
