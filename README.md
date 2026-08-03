# Faily
Local neural-net audio framework — TTS, voice cloning, and SFX. Built for sound design, game development, and digital media production.

---

## Tabs

| Tab | Purpose |
|---|---|
| **CHARACTERS** | Browse and manage voice character library; train Piper TTS models |
| **CLONE** | Upload reference audio, run one-shot voice cloning, build characters |
| **SPEAK** | Load a character and generate audio — expression engine + voice conversion, one-shot, or trained Piper model |
| **EDIT** | Trim, pitch-shift, speed-adjust, and mix generated clips |
| **DAW** | 3-track Web Audio mixer; queue clips from any tab |
| **FOLEY** | Generate sound effects from text prompts (AudioLDM2, Stable Audio, AudioGen) |

---

## System Requirements

| Requirement | Minimum | Notes |
|---|---|---|
| OS | Windows 11 | WSL2 used for dev only |
| GPU | NVIDIA RTX (any) | cu128 required for Blackwell (RTX 50xx series) |
| VRAM | 8 GB | 16 GB recommended for multi-model workflows |
| Python | 3.10+ | 3.12 or 3.14 recommended; 3.11 also needed separately for Piper |
| CUDA | 12.8 | For cu128 PyTorch |
| Git | any | Required for GitHub package installs |
| espeak-ng | any | Required only for Piper training — see below |

---

## Installation

### Step 1 — Run the setup script

```bat
scripts\setup.bat
```

This script does everything below automatically. Read on if you need to install manually or troubleshoot.

---

### Manual install (if setup.bat fails)

#### 1. Create virtual environment

```bat
py -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
```

#### 2. PyTorch cu128 (Blackwell RTX 50xx)

```bat
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
```

> **Other GPU types:** Replace `cu128` with `cu118`, `cu121`, or `cpu` as appropriate.
> See https://pytorch.org/get-started/locally/

#### 3. Core dependencies

```bat
pip install -e .
```

This installs: `nicegui`, `transformers`, `diffusers`, `accelerate`, `soundfile`, `numpy`, `scipy`, `speechbrain`, `huggingface_hub`.

#### 4. Voice cloning backends

```bat
pip install coqui-tts          # XTTS v2 + FreeVC
pip install f5-tts             # F5-TTS (flow-matching diffusion)
pip install chatterbox-tts     # Chatterbox (CFG-guided)
```

#### 5. Expression engines (SPEAK tab, stage 1)

```bat
pip install kokoro             # Kokoro multi-voice TTS
pip install parler-tts         # Parler-TTS (description-driven style)
```

MeloTTS must be installed `--no-deps` on Windows + Python 3.14 — its pinned `tokenizers` version has no cp314 wheel (needs Rust), and `mecab-python3`/`fugashi` require MeCab C headers (not in the standard Windows MeCab installer). Install the current `tokenizers` first so it's satisfied, then add English-only runtime deps manually:

```bat
pip install tokenizers
pip install --no-deps "git+https://github.com/myshell-ai/MeloTTS.git"
pip install librosa inflect langdetect
python -c "import nltk; nltk.download('averaged_perceptron_tagger_eng')"
```

> This gives EN-US/BR/AU/INDIA/Default accents. ES/FR/ZH/JP/KR need `mecab-python3` which requires MeCab dev headers — not in the standard Windows MeCab installer.

#### 6. OpenVoice v2 (SPEAK tab, voice conversion stage 2)

Install `--no-deps` for the same tokenizers pin reason:

```bat
pip install --no-deps "git+https://github.com/myshell-ai/OpenVoice.git"
pip install wavmark
```

> The OpenVoice converter checkpoint (~200 MB) downloads automatically from HuggingFace on first use.

#### 7. Windows environment variable

```bat
setx HF_HUB_DISABLE_SYMLINKS_WARNING 1
```

---

### Step 2 — Piper TTS training (optional)

Piper lets you train a character-specific `.onnx` TTS model from their ref audio clips. This runs in a **separate Python 3.11 venv** to avoid dependency conflicts with the main environment.

**Prerequisites:**
- Python 3.11 — install from https://www.python.org/downloads/release/python-3119/
  - Check "Add python.exe to PATH" during install
- espeak-ng — install from https://github.com/espeak-ng/espeak-ng/releases
  - Use the Windows `.msi` installer (runtime only — no headers needed)

**Run the Piper setup script:**

```bat
scripts\setup_piper.bat
```

This script:
1. Creates `piper_venv/` with Python 3.11
2. Installs `piper-tts` (inference binary)
3. Installs `phonemizer` (pure-Python espeak-ng wrapper — no C compilation)
4. Installs `piper-train --no-deps` + its dependencies manually (skips the uninstallable `piper-phonemize` C extension)
5. Installs `pytorch-lightning~=1.7.0`, `librosa`, `cython`
6. Force-installs torch cu128 (replaces the default torch 1.13 that piper-train pulls)
7. Downloads the `en_US-lessac-medium` base checkpoint (~400 MB) and config

> The `piper_phonemize` compatibility shim is written automatically into the piper venv on the first training run — no manual action needed.

---

## Running

```bat
.venv\Scripts\python main.py
```

Opens at `http://localhost:7842`.

**Native desktop window (no browser):**
```bat
set FAILY_NATIVE=1
.venv\Scripts\python main.py
```

**Kill a hung server:**
```bat
for /f "tokens=5" %a in ('netstat -aon ^| findstr :7842') do taskkill /F /PID %a
```

---

## Backend Reference

### CLONE / ONE SHOT — voice cloning backends

| Backend | Package | Model size | Notes |
|---|---|---|---|
| **SpeechT5** | `transformers` + `speechbrain` | ~300 MB | Fastest; weakest speaker identity |
| **XTTS v2** | `coqui-tts` | ~1.8 GB | Best all-rounder; slow first load |
| **F5-TTS** | `f5-tts` | ~1.2 GB | Highest fidelity with good transcript |
| **Chatterbox** | `chatterbox-tts` | ~750 MB | Best for expressive/emotional range |

All models download automatically from HuggingFace on first use.

---

### SPEAK — Expression engines (stage 1: text → expressive audio)

The expression stage generates intermediate audio in the *style* you describe. Voice conversion then transfers the character's voice onto it.

| Engine | Package | Style control | Notes |
|---|---|---|---|
| **Parler-TTS** | `parler-tts` | Free-text description | "cold fury, slow and deliberate" — expressive but inconsistent |
| **Kokoro** | `kokoro` | Voice name (e.g. `af_heart`, `am_adam`) | Fast; best for short punchy lines |
| **MeloTTS** | `MeloTTS` (GitHub) | Accent dropdown | EN-US/BR/AU/INDIA; very consistent; pairs well with OpenVoice VC. ES/FR/ZH/JP/KR require `mecab-python3` + system MeCab headers — not installable on stock Windows. |

---

### SPEAK — Voice conversion backends (stage 2: expressive audio → character voice)

| Backend | Package | Notes |
|---|---|---|
| **FreeVC** | `coqui-tts` | Fast; best for short clips (<4s) |
| **OpenVoice v2** | `OpenVoice` (GitHub) | Better prosody and naturalness on longer inputs; ~200 MB checkpoint auto-downloads |

---

### CHARACTER — Piper TTS training

Trains a character-specific `.onnx` TTS model from their reference audio clips. Once trained, generating audio in the CHARACTER sub-tab requires no reference clip and produces highly consistent results.

**Requirements:**
- At least 2 ref clips with transcripts (add transcripts via the EDIT button on each clip in the CHARACTERS tab)
- Piper setup complete (see above)

**Training flow:**
1. CHARACTERS tab → select character → TRAIN VOICE
2. A log window shows live training output
3. Exports to `outputs/characters/{name}/piper.onnx` when done
4. Character then appears in the SPEAK → CHARACTER sub-tab

---

### FOLEY — sound effect models

| Model | HuggingFace ID | Notes |
|---|---|---|
| AudioLDM2 | `cvssp/audioldm2` | Default; general purpose |
| AudioLDM2 Large | `cvssp/audioldm2-large` | Slower, higher quality |
| Tango 2 | `declare-lab/tango2` | Strong at environmental sounds |
| AudioGen Medium | `facebook/audiogen-medium` | Fast; good for short effects |
| AudioGen Large | `facebook/audiogen-large` | Better quality |
| Stable Audio Open | `stabilityai/stable-audio-open-1.0` | **Requires HuggingFace login and gated model access** — run `huggingface-cli login` first |

All FOLEY models download automatically on first use (~1–4 GB each).

---

## Models downloaded automatically

| Model | Size | Trigger |
|---|---|---|
| SpeechT5 TTS + HifiGan | ~300 MB | First SpeechT5 generation |
| SpeechBrain x-vector encoder | ~100 MB | First SpeechT5 generation |
| XTTS v2 | ~1.8 GB | First XTTS generation |
| F5-TTS | ~1.2 GB | First F5 generation |
| Chatterbox | ~750 MB | First Chatterbox generation |
| Parler-TTS mini v1.1 | ~900 MB | First Parler-TTS expression pass |
| Kokoro pipeline | ~400 MB | First Kokoro generation |
| MeloTTS | ~200 MB | First MeloTTS expression pass |
| OpenVoice v2 converter | ~200 MB | First OpenVoice voice conversion |
| AudioLDM2 | ~1.5 GB | First FOLEY generation with this model |
| Piper base checkpoint | ~400 MB | `scripts\setup_piper.bat` (manual) |

> Models are cached under `models/` (VC) and `models/sfx/` (FOLEY). Piper checkpoint goes to `piper_checkpoints/`.

---

## Environment variables

| Variable | Value | Purpose |
|---|---|---|
| `HF_HUB_DISABLE_SYMLINKS_WARNING` | `1` | Suppress Windows symlink warnings from HuggingFace |
| `HF_TOKEN` | your token | Required for gated models (Stable Audio Open) |
| `FAILY_NATIVE` | `1` | Run in native desktop window instead of browser |

---

## SPEAK pipeline (two-stage)

The SPEAK → EXPRESSION sub-tab separates *who is speaking* from *how they sound*:

1. **Expression stage** — A style model (Parler-TTS, Kokoro, or MeloTTS) generates intermediate audio from the line + a style description or voice name. This stage has no knowledge of the character's voice.
2. **Voice conversion stage** — FreeVC or OpenVoice v2 takes that expressive audio and maps the character's voice onto it using their reference clips.

This means the two layers improve independently: better style prompts improve stage 1; more approved reference clips in the character's library improve stage 2.

---

## Character voice library (roadmap)

| Stage | Voice conversion | Reference requirement |
|---|---|---|
| **Now** | FreeVC24 | Single reference clip |
| **Now** | OpenVoice v2 | Multi-reference; better on longer output |
| **Future** | RVC (trained model) | Train from all approved clips — sharpens with every good generation |

---

## Clip workflow

Generated clips are named `{character}_{style_word}_{text_word}_{NNN}.wav` (e.g. `obi_wan_cold_hello_001.wav`). Each clip in the history panel has actions:

- **+** — copy to the character's `ref_clips/` pool (feeds future VC training)
- **♥** — copy to `favorites/` (visible in CHARACTERS tab)
- **↓** — copy to system Downloads folder
- **tune** icon — send to EDIT tab
- **music** icon — queue in DAW tab

Clips not acted on remain in `outputs/vc/` (master dump).

---

## Platform
Tested on Python 3.14, PyTorch 2.11.0+cu128, RTX 5070 Ti (Blackwell sm_120), Windows 11.

All Rights Reserved, Sky Vercauteren 2026
