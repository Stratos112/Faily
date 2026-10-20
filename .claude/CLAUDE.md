# Faily — Claude Context

## What this is
Local audio app (NiceGUI) for TTS, voice cloning, and sound effects.
Primary deployment is a Docker container (single image, two venvs — installed by `scripts/setup_linux.sh`, which the `Dockerfile` runs phase by phase) on the user's Windows 11 / RTX 5070 Ti (Blackwell) machine, GPU passed through via Docker Desktop + NVIDIA Container Toolkit. Dev happens in WSL2 at `/workspaces/projects/Faily`.

History: Faily originally ran natively on Windows (setup.bat/setup_piper.bat, removed in the Docker port). The `monotonic_align` MSVC crash saga in git history is from that era and was part of the motivation for moving to Docker; it does not reproduce on Linux.

## Running
```bash
docker compose up --build   # starts on http://localhost:7842
bash scripts/setup_linux.sh # bare-metal / dev-box install (all phases; or name phases: system main piper app data)
bash scripts/test_faily.sh  # full test run + clean shutdown (run against the container, or inside it)
```
Container restart/stop: `docker compose down` / `docker compose restart` (replaces the old Windows `taskkill`/`netstat` dance).

Volumes (persist across rebuilds): `./outputs`, `./models`, `./piper_checkpoints`, `./config`. The HuggingFace cache lives in `models/hf` (`HF_HOME`, set in `faily/__init__.py`) and Coqui's in `models/vc/coqui`, so they persist in both the dev workspace and the container. `entrypoint.sh` runs `setup_linux.sh data` — the two setup steps that write into those volume-mounted dirs (Seed-VC repo clone, Piper base checkpoint download) at container start, since anything baked into those paths at image-build time would just be shadowed by the bind mounts.

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

**Model loading** — always via `manager.load(key, loader_fn)`. Never import model libs at module level; always inside the loader lambda/function. The manager evicts least-recently-used models before a load until `FAILY_VRAM_RESERVE_GB` (default 6) is free, and retries once after unloading everything on CUDA OOM. Under WSL2/Docker Desktop the driver spills to shared system RAM instead of raising OOM, so the pre-load eviction is what actually protects performance there.

**Auth** — optional HTTP Basic auth via `FAILY_PASSWORD` (`_BasicAuth` ASGI middleware in app.py, HTTP scope only). Off by default.

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
- Piper training status/stop: `train()` writes `outputs/characters/<name>/piper_training.json`; the training dialog and header chips poll it (`ui/tabs/piper_training_dialog.py`). "Stop early" drops `piper_train/STOP_REQUEST`, honored by a callback that `scripts/patch_piper_train_stop_request.py` injects into piper_train (saves an epoch-exact checkpoint, then stops); without the patch it falls back to SIGTERM. Learning rate: piper_train has no LR flag and Lightning restores the checkpoint's optimizer state on resume, so `scripts/patch_piper_train_learning_rate.py` adds a callback that applies `FAILY_PIPER_LR` after the restore (UI field on the TRAINING tab, persisted as `piper_learning_rate` in settings, default 5e-5; lessac's checkpoint sits at ~1.5e-4). Match options: `scripts/patch_piper_train_freeze.py` honors `FAILY_PIPER_FREEZE` (comma list of generator submodules, e.g. `enc_p,dp`); the TRAINING tab has MATCH toggles per aspect (timbre → dec/flow/enc_q, rhythm → dp, pronunciation → enc_p; `MATCH_ASPECTS` in modules/piper.py, settings `piper_match_*`, default timbre=reference, rhythm/pronunciation=base) — an aspect set to Base Model is frozen. `piper_use_padding` toggles generated buffer clips. Pre-training data checks live in `modules/cleanup.py` + `ui/tabs/cleanup_dialog.py` (volume / trim / transcript passes, Whisper via `modules/asr.py`; originals → `refs_original/`, deletes → `refs_trash/`); checkpoint audition rows show a Whisper pronunciation score (`modules/pronunciation.py`). Unfrozen fine-tuning on a small different-speaker set wrecks pronunciation (measured via Whisper WER on unseen sentences). Progress: Lightning prints no bar when piped, so `patch_piper_train_progress.py` writes `piper_train/progress.json` and train() polls it. Models: every saved Piper model lives in `outputs/characters/<name>/models/<id>/` (model.onnx, checkpoint.ckpt, meta.json, cached pronunciation score) — registry in `core/piper_models.py`, MODELS tab in `ui/tabs/models_panel.py`. The audition list's SAVE / SAVE & USE call `piper.promote_candidate()`; the config's `piper_model` points at the active model. Training can resume from a model's checkpoint (`resume_model`) and optionally replace it (`replace_id`); each run's settings are recorded in the status file's `run` and copied into meta.json. A pre-registry `piper.onnx` is adopted on first view (checkpoint matched by file time, base voice inferred). Import: `modules/slicer.py` splits a recording at pauses; `modules/importer.py` stages the slices in a hidden `_stage_*` character; `ui/tabs/import_dialog.py` + `cleanup_dialog(staging=...)` run the spell-check on them (Whisper transcript pre-filled) before `commit_stage()` adds them to a character — reachable from the CLONE tab and the REFERENCES tab. The loss graph reads TensorBoard events from `piper_train/lightning_logs/`
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
