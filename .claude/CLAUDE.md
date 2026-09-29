# Faily — Claude Context

## What this is
Local audio app (NiceGUI) for TTS, voice cloning, and sound effects.
Primary deployment is a Docker container (single image, two venvs — installed by `scripts/setup_linux.sh`, which the `Dockerfile` runs phase by phase) on the user's Windows 11 / RTX 5070 Ti (Blackwell) machine, GPU passed through via Docker Desktop + NVIDIA Container Toolkit. Dev happens in WSL2 at `/workspaces/projects/Faily`.

Legacy: `scripts/setup.bat` / `scripts/setup_piper.bat` still exist for running natively on Windows (two local venvs, no Docker) but are no longer the primary path — native Windows was where the `monotonic_align` MSVC crash saga (see git history) came from, part of the motivation for moving to Docker.

## Running
```bash
docker compose up --build   # starts on http://localhost:7842
bash scripts/setup_linux.sh # bare-metal / dev-box install (all phases; or name phases: system main piper app data)
bash scripts/test_faily.sh  # full test run + clean shutdown (run against the container, or inside it)
```
Container restart/stop: `docker compose down` / `docker compose restart` (replaces the old Windows `taskkill`/`netstat` dance).

Volumes (persist across rebuilds): `./outputs`, `./models`, `./piper_checkpoints`, plus a named `hf-cache` volume for the HuggingFace cache. `entrypoint.sh` runs `setup_linux.sh data` — the two setup steps that write into those volume-mounted dirs (Seed-VC repo clone, Piper base checkpoint download) at container start, since anything baked into those paths at image-build time would just be shadowed by the bind mounts.

## Architecture

```
faily/
  core/
    model_manager.py   # ModelManager singleton — lazy load/cache/unload models
    characters.py      # Character storage (outputs/characters/{name}/config.json)
  modules/
    tts.py             # Bark / MMS-TTS via HF pipeline
    vc.py              # Voice cloning — BACKENDS dict, generate(), torchaudio patches
    foley.py           # AudioLDM2 SFX generation
  ui/
    app.py             # NiceGUI app entry, tabs: CLONE / TUNE / TTS / FOLEY
    components.py      # output_panel(), section_label(), show_error()
    tabs/
      vc_tab.py        # CLONE: ref audio upload, character save, voice preview
      tune_tab.py      # TUNE: character picker, expression sculpting, sub-character save
      tts_tab.py       # TTS: Bark/MMS narration, legacy clone-voice picker
      foley_tab.py     # FOLEY: SFX generation
```

## Key patterns

**Model loading** — always via `manager.load(key, loader_fn)`. Never import model libs at module level; always inside the loader lambda/function.

**vc.py BACKENDS** — dict drives both UI dropdown and dispatch. Add a new backend by adding an entry + loader + generate function.

**Characters** — stored in `outputs/characters/{name}/`. Base characters have `ref_audio`. Sub-characters have `parent` + expression params (`backend`, `param1`, `param2`, `speed`, `style_prompt`). `get_ref_path(name)` resolves sub→parent transparently.

**torchaudio patch** — `_patch_torchaudio()` in vc.py replaces `torchaudio.load` globally with a soundfile implementation (torchcodec DLLs missing on user's Windows RTX 5070 Ti system). `_patch_ffmpeg_read()` does the same for transformers' ASR pipeline.

**Progress reporting** — functions accept `progress_ref: list[float]` and write 0.0–1.0 into it. UI polls via `ui.timer(0.15, ...)`.

## Platform notes
- Container: Ubuntu 24.04, Python 3.14 (main app venv) + Python 3.11 (piper_venv), both uv-managed (in `.python/`), PyTorch cu128, RTX 5070 Ti (Blackwell sm_120) passed through from the Windows host via Docker Desktop + NVIDIA Container Toolkit
- Main venv is 3.14 deliberately — chatterbox-tts pins `torch==2.6.0` on Python < 3.14 (no Blackwell sm_120 kernels); only on 3.14 does it accept torch>=2.9. `setup_linux.sh` re-asserts cu128 torch after all backend installs anyway. parler-tts is installed `--no-deps` (it pins transformers==4.46.1; vc.py's patches target 5.x)
- `torchaudio.load` still patched to soundfile (`_patch_torchaudio()` in vc.py) — harmless/still fine on Linux even though torchcodec is more likely to actually work here than it was on Windows
- transformers 5.x → `isin_mps_friendly` patched back onto `pytorch_utils`
- `piper_train`'s `monotonic_align` Cython extension is built from stock source on Linux (gcc) and survives real GPU training (verified 2026-09-29) — the MSVC crash was Windows-only. `scripts/patch_piper_train_monotonic_align_pure_python.py` remains as a fallback
- torchcodec: the cu128-index build (not PyPI's CUDA-13 one) + `nvidia-npp-cu12`, loaded via `_preload_npp()` in model_manager.py (the wheel has no rpath to NPP). Faily's own code avoids torchcodec anyway (soundfile everywhere; `transcribe_ref` passes decoded samples)
- Seed-VC vendors its own BigVGAN; `_load_seedvc` shims its `_from_pretrained` to default proxies/resume_download (no site-packages patching)
- `FAILY_NATIVE` env var must stay unset in the container — that path launches a pywebview native window, which has no display to attach to in a headless container. Server mode (the default) is what serves the browser UI at `:7842`

## Output dirs
```
outputs/tts/           # TTS generations
outputs/vc/            # Voice clone generations
outputs/vc/refs/       # Reference audio samples (legacy)
outputs/characters/    # Saved characters
outputs/sfx/           # Foley/SFX
```
