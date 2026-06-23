# Faily — Claude Context

## What this is
Local desktop audio app (NiceGUI) for TTS, voice cloning, and sound effects.
Runs on Windows (user's machine at `E:\PROGRAMS\Faily`), dev happens in WSL2 at `/workspaces/projects/Faily`.

## Running
```bash
python main.py          # starts on http://localhost:7842
bash scripts/test_faily.sh   # full test run + clean shutdown
```
Windows kill: `for /f "tokens=5" %a in ('netstat -aon ^| findstr :7842') do taskkill /F /PID %a`

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
- Python 3.14, PyTorch 2.11.0+cu128, Windows 11, RTX 5070 Ti (Blackwell sm_120)
- torchcodec doesn't work → torchaudio.load patched to use soundfile
- speechbrain LazyModule patched for Python 3.14 inspect changes
- transformers 5.x → `isin_mps_friendly` patched back onto `pytorch_utils`
- cu128 required for Blackwell GPU support

## Output dirs
```
outputs/tts/           # TTS generations
outputs/vc/            # Voice clone generations
outputs/vc/refs/       # Reference audio samples (legacy)
outputs/characters/    # Saved characters
outputs/sfx/           # Foley/SFX
```
