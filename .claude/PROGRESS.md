# Server migration: progress notes

These notes live in the repo because the dev container doesn't persist `~/.claude`, so memory is wiped on rebuild.

## Goal
Run Faily headless on a Linux GPU server in Docker instead of natively on Windows.
For now, the dev container itself (Debian 12, RTX 5070 Ti via `--gpus=all`) stands in for "the docker box".

## State as of 2026-09-28
- The Dockerfile, compose file and entrypoint are written (commit b5da4d5) but have never been built. This dev container has no docker CLI.
- Nothing has been built or tested in this box yet; the user is using the GPU.

## Findings: what blocks running here
1. **No shared install path.** The Dockerfile targets Ubuntu 24.04 + deadsnakes. This box is Debian 12 with Python 3.11 only (no 3.12). Plan: move the install into `scripts/setup_linux.sh` and have the Dockerfile `RUN` it, so the dev box and the image install the same way. Get Python 3.12 via `uv`, or run the main venv on 3.11.
2. **Missing system packages here:** `espeak-ng` (Piper phonemization) and `ffmpeg`. The Dockerfile doesn't install ffmpeg either.
3. **The existing `piper_venv/` has torch 1.13.1 (cu117)** from the Aug 3 WSL setup and can't run on sm_120. It needs the cu128 force-reinstall; `setup_piper.sh` skips that step and still says "use Windows for GPU".
4. **No main `.venv`** exists here yet.
5. **Coqui TOS prompt:** XTTS v2 and FreeVC call `input()` for license acceptance on first load. A headless server has no stdin, so the load hangs or fails. Set `COQUI_TOS_AGREED=1`.
6. **Coqui model cache** (`~/.local/share/tts`) isn't on any volume in the compose file, so XTTS (~2 GB) would re-download on every rebuild.
7. **Download buttons copy files to the server's `~/Downloads`** (components.py `_download_local`, edit_tab `_download`, daw_tab `_mix_download`). On a server that's invisible to the user. Switch to `ui.download()`, which sends the file to the browser, and drop the "download location" setting.
8. **Settings and HF token are stored at the repo root** (`faily_settings.json`, `hf_token.txt`). In the image, `/app` isn't a volume, so both are lost on rebuild. Move them into a persisted dir, and have `get_hf_token()` fall back to the `HF_TOKEN` env var (the README already claims it does).
9. **No auth.** The app binds 0.0.0.0:7842 with no login. That's fine on a LAN; anything exposed further needs a reverse proxy with auth, or NiceGUI's storage_secret + a login page.
10. **RAM:** this box has 15 GB of system RAM, and `ModelManager` never evicts models. Loading several VC backends in one session can OOM.

## Cleanup (not blocking)
- Windows-only code in `piper.py`: the `_IS_WIN` branches, the eSpeak `C:\Program Files` lookup, the `phonemizer` vs `piper_train` diagnose split, and the WindowsPath checkpoint hack.
- Crash diagnostics in `piper.py`'s training env: `CUDA_LAUNCH_BLOCKING=1` slows training, and `PYTHONFAULTHANDLER` was added for the same crash. Remove both once training works on Linux.
- The `monotonic_align` patch scripts (serial/noexcept/nan/bounds). Keep only the pure-Python fallback, and only if the stock build crashes.
- `setup.bat`, `setup_piper.bat`, and the `.venv\Scripts` docstrings in the patch scripts.
- `test_faily.sh`: it checks for `CLAUDE.md` at the repo root (it's in `.claude/`, so that check always fails), uses the system `python3` instead of the venv, and its skip message says "run on Windows".
- README setup section is Windows-only.

## Plan / TODO
Tick items off here as they land; this file is the source of truth, not ~/.claude memory.

### Phase 1: code fixes that don't need the GPU
- [x] 1a. Coqui: set `COQUI_TOS_AGREED=1`, and point `TTS_HOME` at `models/coqui` so the XTTS/FreeVC cache is on the models volume
- [x] 1b. Download buttons send files to the browser (`ui.download.file`) instead of copying to the server's ~/Downloads; drop the download-location setting
- [x] 1c. Move settings and the HF token into a persisted `config/` dir (`FAILY_CONFIG_DIR` overrides it, and old root-level files migrate over); `get_hf_token()` falls back to the `HF_TOKEN` env var; add `./config` to the compose volumes
- [x] 1d. `piper.py`: remove the Windows branches, eSpeak Windows paths and `CUDA_LAUNCH_BLOCKING`
- [x] 1e. `test_faily.sh`: fix the CLAUDE.md path, use the venv python, drop the Windows messaging

Phase 1 notes (2026-09-28):
- DAW "MIX & DOWNLOAD" now also keeps a copy in `outputs/mix/`, since the mix needs a server-side file to serve.
- The `PYTHONFAULTHANDLER` training env was kept (free, and useful if monotonic_align segfaults on Linux); only `CUDA_LAUNCH_BLOCKING` was removed.
- pyproject/Dockerfile now pin `nicegui>=3.0.0`: the upload handlers already use the 3.x `e.file` API, and `ui.download.file` needs it too.
- test_faily.sh: the stubbed checks use the venv python. The BACKENDS check still fails until a venv with numpy exists; that predates Phase 1.
- It also works without `fuser` now (this box has no fuser/lsof/ss).

### Phase 2: one install path (no GPU needed to write, only to run)
- [x] 2a. `scripts/setup_linux.sh`: system packages (espeak-ng, ffmpeg, build tools), main venv, piper venv (cu128 torch), patches. Idempotent.
- [x] 2b. The Dockerfile runs setup_linux.sh instead of duplicating the steps
- [x] 2c. Main venv Python: **3.14** (uv-managed); piper venv 3.11 (uv-managed)

Phase 2 notes (2026-09-28):
- **Why 3.14, not 3.12:** chatterbox-tts 0.1.7 pins `torch==2.6.0` on Python < 3.14 (torch>=2.9 on 3.14+). torch 2.6 has no sm_120 kernels, so the Dockerfile's old 3.12 choice would have silently downgraded torch. 3.14 also matches the known-working Windows setup.
- parler-tts pins `transformers==4.46.1` exactly and is now installed `--no-deps` plus its own small deps. chatterbox pins transformers 5.2.0, and vc.py's parler patches target 5.x.
- Expected `pip check` noise: gradio (chatterbox ==6.8.0 vs f5 >=6.15), diffusers (chatterbox ==0.29.0), parler's transformers pin. Faily doesn't use gradio.
- Untested risk: cp314 Linux wheels for faster-whisper/ctranslate2, bitsandbytes, s3tokenizer, spacy-pkuseg. `pip_soft` warns per package instead of aborting.
- Script phases: system | main | piper | app | data. The Dockerfile runs system/main/piper/app as separate layers (with a pip cache mount); entrypoint.sh runs `data`.
- `.dockerignore` now excludes `hf_token.txt`/`config/`. Before this, a saved HF token would have been baked into the image.
- The dev box's existing piper_venv passes the "complete" check, so `piper` only upgrades its torch 1.13.1 → cu128.
- Tested without installing anything: bash -n, phase dispatch, and pip_soft's retry/warn logic with a fake pip. Nothing actually installed yet.

### Phase 3: install + test in this box (NEEDS THE GPU FREE; ask the user first)
- [x] 3a. `bash scripts/setup_linux.sh`: done 2026-09-28. main=.venv py3.14.7 torch 2.11.0+cu128; piper_venv torch 2.11.0+cu128
- [x] 3b. test_faily.sh passes (static + live): 42/42
- [x] 3c. One smoke generation per backend: all pass (see results table below)
- [x] 3d. Piper training run: YES. Stock gcc monotonic_align survives a real GPU epoch on Linux; the Windows crash doesn't reproduce

Phase 3 notes (2026-09-28):
- **Old .venv trap:** a Python 3.11 system-python `.venv` from Aug was silently reused on the first run. It had torch 2.12.1+cu130 next to torchaudio 2.11.0+cu128, and torchaudio failed to import. setup_linux.sh now moves a wrong-version venv aside (`.venv.old-pyXY`); the old one is at `.venv.old-py311` (6.6 GB, safe to delete). The torch check now accepts CUDA >= 12.8 and requires matching torch/torchaudio versions, with a clean reinstall as fallback.
- **Shared deps clobbered by backend pins**, now repaired by a step in setup_linux.sh:
  - chatterbox's gradio==6.8.0 dragged starlette to 0.52 under nicegui (needs >=1.3.1).
  - descript-audiotools (Seed-VC deps) dragged protobuf to 3.19.6, breaking onnx/wandb/onnxruntime.
- **bigvgan/hub_mixin patches were wrong-headed.** patch_bigvgan.py silently no-op'd, because bigvgan 2.4.1 moved `_from_pretrained` to bigvgan/bigvgan.py; and nothing imports pip `bigvgan` anyway. Seed-VC uses its OWN vendored `modules/bigvgan`. patch_hub_mixin.py globally injected proxies/resume_download into huggingface_hub, which would crash any PyTorchModelHubMixin model without its own `_from_pretrained`. Both are replaced by a runtime shim in `vc._load_seedvc`, and the scripts are no longer run (delete them in 4a).
- Stock gcc monotonic_align imports and runs a toy maximum_path in piper_venv (a real training run is still 3d).
- `ui.run(show=...)` now only opens a browser in native mode; server mode used to try to launch one on the host.
- test_faily.sh static-asset URL fixed for NiceGUI 3's versioned `/_nicegui/<ver>/static`.
- Remaining `pip check` noise is unused paths only: gradio, the audiotools protobuf pins, parler's transformers pin, and openvoice's stale pins.

Phase 3c/3d results (2026-09-29; synthetic MMS reference voice; times include first-run model downloads):

| case | result | time | peak VRAM |
|---|---|---|---|
| MMS-TTS | OK | 6s | 0.2G |
| SpeechT5 | OK | 55s | 0.7G |
| XTTS v2 | OK (cached in models/vc/coqui, no TOS prompt) | 152s | 1.8G |
| F5-TTS | OK | 84s | 1.9G |
| transcribe_ref (Whisper) | OK after fix | 10s | 1.7G |
| Chatterbox | OK | 141s | 3.2G |
| AudioLDM2 | OK | 15s | 2.3G |
| Piper infer (lessac preview) | OK | 6s | n/a |
| Parler + FreeVC | OK | 386s | 4.2G |
| Parler + OpenVoice | OK | 30s | 4.1G |
| Parler + Seed-VC | OK (runtime BigVGAN shim) | 189s | 5.7G |
| Piper training, 1 epoch from lessac | OK, no crash | ~1 min | n/a |

Fixes found by 3c/3d:
- **torchcodec:** pip pulled 0.16.0 (CUDA 13 build, needs libnvrtc.so.13) and transformers 5 decodes ASR file paths through it, bypassing `_patch_ffmpeg_read`. Three fixes:
  - `transcribe_ref` now passes decoded samples.
  - setup pins torchcodec to the cu128 index (0.11.1) and installs `nvidia-npp-cu12`.
  - `model_manager._preload_npp()` loads NPP RTLD_GLOBAL, because the wheel has no rpath to it.
- **Piper epoch off-by-one (pre-existing, not Linux-specific):** a checkpoint's `epoch` is the epoch that just finished, so resume starts at +1. `max_epochs=1` trained 0 epochs and 1000 trained 999. Fixed in piper.py.
- **Seed-VC wrote its 1.4 GB weight cache to `./checkpoints`** (cwd-relative; /app/checkpoints in the container, outside every volume). `_load_seedvc` now redirects it to `models/vc/seed-vc-checkpoints`. **OpenVoice's `get_se` left scratch in `./processed`** forever; it now gets a temp dir that's cleaned up. Both pipelines were re-verified, and nothing is written to the repo root anymore.
- Open question for the user: `outputs/` and `piper_checkpoints/*.onnx` show as untracked. Should they be gitignored? They're user data/downloads on volumes.
- `.venv.old-py311/` (6.6 GB) is the moved-aside old venv; delete it once happy.
- The dev box's HF cache (`~/.cache/huggingface`: F5, Chatterbox, Parler, Whisper, ...) is NOT persisted across dev-container rebuilds. In Docker it's the `hf-cache` volume.

### Phase 4: cleanup + hardening
- [ ] 4a. Delete setup.bat / setup_piper.bat / setup_piper.sh (superseded by setup_linux.sh) / monotonic_align patch-trail scripts (keep the pure-python fallback) once 3d passes
- [ ] 4b. Rewrite README setup for Docker/Linux; update the test-faily skill and CLAUDE.md
- [ ] 4c. Model eviction in ModelManager (RAM/VRAM limit), e.g. an LRU or unload-before-load for big models
- [ ] 4d. Auth or reverse-proxy guidance if the server is exposed beyond the LAN
