# Faily
Local neural-net audio framework — TTS, voice cloning, and SFX. Built for sound design, game development, and digital media production.

## Tabs
| Tab | Purpose |
|---|---|
| **CLONE** | Upload reference audio, create reusable voice characters |
| **TUNE** | Load a character, sculpt expression/emotion, save as sub-character |
| **TTS** | Narrate text with Bark or MMS-TTS (fixed-voice models) |
| **FOLEY** | Generate sound effects from text prompts via AudioLDM2 |

## Voice cloning backends (CLONE / TUNE)
- **SpeechT5** — Microsoft, x-vector speaker embeddings, CPU-friendly
- **XTTS v2** — Coqui AI, zero-shot cloning, cross-attention conditioning
- **F5-TTS** — Flow-matching diffusion, quality scales with steps
- **Chatterbox** — Resemble AI, CFG-guided, best for expression control

## Setup

```bash
pip install -e .
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install coqui-tts f5-tts chatterbox-tts
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
