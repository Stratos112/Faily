# Faily

Local neural-net audio framework — TTS, voice cloning, and SFX. Built for sound design, game development, and digital media production.

---

## Tools

One pipeline: create a character, generate its voice, layer in sound, polish, and feed the result back in.

| Tab | What it does |
|---|---|
| **CHARACTERS** | Character library — browse/manage everyone you've built, train a character-specific Piper voice model |
| **CLONE** | Zero-shot voice cloning from a short reference clip — where new characters start |
| **SPEAK** | Generate lines — expression engine (style-driven delivery + voice conversion) or a trained Piper model, whichever the character has |
| **FOLEY** | Text-prompted sound effect generation |
| **EDIT** | Trim, pitch-shift, speed-adjust, denoise, and mix any generated clip |
| **DAW** | 3-track mixer to assemble clips from any tab into a final piece |

---

## Setup

**Requirements**
- Windows 11 (WSL2 used for dev only)
- NVIDIA RTX GPU, 8 GB+ VRAM (cu128 required for Blackwell / RTX 50xx)
- Python 3.10+ (3.12 or 3.14 recommended)
- Git
- Python 3.11 + espeak-ng — only needed for Piper training (optional)

**Install**
1. Run `scripts\setup.bat` — installs everything: venv, PyTorch cu128, core deps, all cloning/expression backends. If it fails partway, open the script and run the remaining `pip install` lines by hand.
2. *(Optional — for training character-specific Piper models)* Install Python 3.11 and espeak-ng, then run `scripts\setup_piper.bat`.

**Run**
```bat
.venv\Scripts\python main.py
```
Opens at `http://localhost:7842`. Set `FAILY_NATIVE=1` to run in a native window instead of a browser.

Kill a hung server:
```bat
for /f "tokens=5" %a in ('netstat -aon ^| findstr :7842') do taskkill /F /PID %a
```

**Environment variables**
- `HF_TOKEN` — required for gated models (Stable Audio Open)
- `HF_HUB_DISABLE_SYMLINKS_WARNING=1` — quiets a Windows-only HuggingFace warning

---

## TODO

- **Expressive Piper output** — Piper has no style/emotion conditioning. Research a backend with native style conditioning (e.g. StyleTTS2) compatible with the torch 2.11+cu128/Blackwell stack, as an alternative to Piper for CHARACTER training.
- **New FOLEY backends** — before adding one, confirm a real Windows/cu128-compatible install path and actually run a generation with it. Tango 2 and AudioGen were already tried and dropped for failing this.
- **Promote-to-ref-pool prominence** — saving a good generation back into a character's reference pool already works from any clip; surface it more (e.g. auto-suggest after favoriting).
- **RVC-trained-model voice conversion** — a lighter-weight trained-model path than a full Piper fine-tune; worth prioritizing if Piper's data requirements (30–200+ clips for good results) prove too heavy for most characters.
- **YouTube URL scraper** — paste a link in CLONE, pull reference audio automatically instead of manual upload.
- **CLONE / CHARACTERS split** — character management (save/delete/list) currently lives partly in CLONE; it belongs in CHARACTERS so CLONE can focus purely on the cloning/auditioning process.

---

All Rights Reserved, Sky Vercauteren 2026
