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

Faily runs headless on a Linux box with an NVIDIA GPU and is used from a browser.

**Requirements**
- NVIDIA GPU, 8 GB+ VRAM (Blackwell / RTX 50xx supported: everything runs on CUDA 12.8 builds)
- ~40 GB of disk for the image/venvs plus downloaded models
- Docker with the NVIDIA Container Toolkit (Docker Desktop on Windows works: WSL2 backend, GPU passthrough), **or** a bare Debian/Ubuntu machine for the direct install

**Run with Docker (recommended)**
```bash
docker compose up --build -d     # first build downloads ~20 GB
```
Open `http://<server>:7842`. Stop with `docker compose down`; logs with `docker compose logs -f`.

Everything worth keeping lives in bind-mounted folders next to `docker-compose.yml`, so rebuilding the image loses nothing:

| Folder | Contents |
|---|---|
| `outputs/` | characters, generations, mixes |
| `models/` | downloaded model weights (HuggingFace cache in `models/hf`, Coqui in `models/vc/coqui`, Seed-VC repo + weights) |
| `piper_checkpoints/` | Piper base voices for training |
| `config/` | settings + saved HF token |

On first start the container clones Seed-VC and downloads the Piper base checkpoint (~800 MB). Models download on first use.

**Run directly (no Docker)**
```bash
bash scripts/setup_linux.sh      # apt packages, uv, both venvs, data; rerunnable
.venv/bin/python main.py
```
The script can also run single phases (`system`, `main`, `piper`, `app`, `data`); see its header. It ends with a summary of any backend that failed to install.

**Tests**
```bash
bash scripts/test_faily.sh       # syntax/import checks + starts the server and probes it
```

**Environment variables**
- `HF_TOKEN`: HuggingFace token for gated models (Stable Audio Open). Alternatively, paste one into Settings (gear icon).
- `FAILY_CONFIG_DIR`: where settings and the saved token live (default `config/`)
- `HF_HOME`: HuggingFace cache location (default `models/hf`)
- `FAILY_PASSWORD` / `FAILY_USER`: turn on a browser login (HTTP Basic auth; user defaults to `faily`). Off when unset.
- `FAILY_VRAM_RESERVE_GB`: before loading a model, Faily unloads the least recently used ones until this much VRAM is free (default `6`). Lower it to keep more models warm; raise it if you run other GPU work alongside.

Set these under `environment:` in `docker-compose.yml`, for example:
```yaml
    environment:
      - FAILY_PASSWORD=change-me
      - HF_TOKEN=hf_...
```

**Security:** By default Faily has no login. Anyone who can reach port 7842 can use it, including uploading files and running GPU jobs. That's fine on a trusted home network. Otherwise, set `FAILY_PASSWORD`. If the server is reachable from the internet, also put it behind a TLS reverse proxy (e.g. Caddy or nginx). Basic auth over plain HTTP sends the password in cleartext.

---

## TODO

- **Expressive Piper output** — Piper has no style/emotion conditioning. Research a backend with native style conditioning (e.g. StyleTTS2) compatible with the torch 2.11+cu128/Blackwell stack, as an alternative to Piper for CHARACTER training.
- **New FOLEY backends** — before adding one, confirm it installs via `scripts/setup_linux.sh` on the cu128 stack and actually run a generation with it. Tango 2 and AudioGen were already tried and dropped for failing this.
- **Promote-to-ref-pool prominence** — saving a good generation back into a character's reference pool already works from any clip; surface it more (e.g. auto-suggest after favoriting).
- **RVC-trained-model voice conversion** — a lighter-weight trained-model path than a full Piper fine-tune; worth prioritizing if Piper's data requirements (30–200+ clips for good results) prove too heavy for most characters.
- **YouTube URL scraper** — paste a link in CLONE, pull reference audio automatically instead of manual upload.
- **CLONE / CHARACTERS split** — character management (save/delete/list) currently lives partly in CLONE; it belongs in CHARACTERS so CLONE can focus purely on the cloning/auditioning process.

---

All Rights Reserved, Sky Vercauteren 2026
