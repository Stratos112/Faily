# Faily
Local neural-net audio framework — TTS, voice cloning, and SFX. Built for sound design, game development, and digital media production.

## Tabs
| Tab | Purpose |
|---|---|
| **CLONE** | Upload reference audio, create reusable voice characters |
| **TUNE** | Load a character, describe their expression in text, generate in their voice |
| **TTS** | Narrate text with Bark or MMS-TTS (fixed-voice models) |
| **FOLEY** | Generate sound effects from text prompts via AudioLDM2 |

## Voice cloning backends (CLONE)
- **SpeechT5** — Microsoft, x-vector speaker embeddings, CPU-friendly
- **XTTS v2** — Coqui AI, zero-shot cloning, cross-attention conditioning
- **F5-TTS** — Flow-matching diffusion, quality scales with steps
- **Chatterbox** — Resemble AI, CFG-guided, best for expression control

## TUNE pipeline (two-stage)
The TUNE tab uses a two-stage pipeline that separates *who is speaking* from *how they're speaking*:

1. **Expression stage** — A text-description model (Parler-TTS or Orpheus TTS) generates speech from the line + a free-text style prompt ("sing-song, happy, silly but serious"). This stage doesn't know or care about the character's voice.
2. **Voice conversion stage** — FreeVC (via Coqui TTS) takes the expressive audio and converts it to the character's voice using their reference audio from the character library.

This separation means the two layers improve independently: better style descriptions improve stage 1; more approved reference audio improves stage 2.

## Character voice library (roadmap)
Characters start with one reference clip from the CLONE tab. Over time, approved generations can be added back to the character's profile, growing their voice library. The stage-2 voice conversion model is designed to improve as the library grows:

| Stage | Voice conversion model | Reference requirement |
|---|---|---|
| **Now** | FreeVC24 | Single reference clip |
| **Growing library** | OpenVoice v2 | Multi-reference, better consistency |
| **Mature character** | RVC (trained) | Train a lightweight voice model from all approved clips — the character gets sharper with every good generation |

Because voice identity (stage 2) and expression (stage 1) are separate, adding a new reference clip to the library improves the character's voice without touching anything else.

## Setup

```bash
pip install -e .
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install coqui-tts f5-tts chatterbox-tts
pip install kokoro  # expression stage (Kokoro + FreeVC pipeline)
```

**Environment:**
```
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
```

**Run:**
```
python main.py
# opens at http://localhost:7842
```

**Native desktop window:**
```
set FAILY_NATIVE=1 && python main.py
```

## Kill a hung server (Windows)
```
for /f "tokens=5" %a in ('netstat -aon ^| findstr :7842') do taskkill /F /PID %a
```

## Platform
Tested on Python 3.14, PyTorch 2.11.0+cu128, RTX 5070 Ti (Blackwell), Windows 11.

All Rights Reserved, Sky Vercauteren 2026
