import soundfile as sf
from datetime import datetime
from pathlib import Path
from faily.core.model_manager import manager

VC_OUTPUT_DIR = Path("outputs/vc")

BACKENDS = {
    "xtts_v2": {
        "label": "XTTS v2",
        "desc": "Coqui AI · Zero-shot · Cross-attention conditioning",
        "param1": {"label": "TEMPERATURE", "tooltip": "Expressiveness. Low = flat and consistent. High = emotive but may wander.", "min": 0.1, "max": 1.0, "step": 0.05, "default": 0.75},
        "param2": {"label": "SPEED", "tooltip": "Speech rate. 1.0 is natural pace.", "min": 0.5, "max": 2.0, "step": 0.05, "default": 1.0},
    },
    "f5_tts": {
        "label": "F5-TTS",
        "desc": "SWC Lab · Flow-matching diffusion · Quality scales with steps",
        "param1": {"label": "STEPS", "tooltip": "Diffusion steps. More = higher quality but slower. 32 is a good balance.", "min": 8, "max": 64, "step": 4, "default": 32},
        "param2": {"label": "SPEED", "tooltip": "Speech rate. 1.0 is natural pace.", "min": 0.5, "max": 2.0, "step": 0.05, "default": 1.0},
    },
    "chatterbox": {
        "label": "Chatterbox",
        "desc": "Resemble AI · CFG-guided · Emotion exaggeration control",
        "param1": {"label": "EXAGGERATION", "tooltip": "Emotional intensity. Low = calm and neutral. High = expressive.", "min": 0.0, "max": 1.0, "step": 0.05, "default": 0.5},
        "param2": {"label": "CFG WEIGHT", "tooltip": "Guidance strength. Higher = more faithful to the reference voice style.", "min": 0.0, "max": 1.0, "step": 0.05, "default": 0.5},
    },
}


def _patch_xtts_transformers():
    # transformers 5.x removed isin_mps_friendly; coqui-tts still imports it
    import torch
    import transformers.pytorch_utils as pu
    if not hasattr(pu, "isin_mps_friendly"):
        pu.isin_mps_friendly = torch.isin


def _load_xtts():
    _patch_xtts_transformers()
    from TTS.api import TTS
    gpu = str(manager.device).startswith("cuda")
    return TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=gpu)


def _load_f5():
    from f5_tts.api import F5TTS
    return F5TTS(device=str(manager.device))


def _load_chatterbox():
    from chatterbox.tts import ChatterboxTTS
    return ChatterboxTTS.from_pretrained(device=str(manager.device))


def _xtts_generate(text, ref_path, out, temperature, speed):
    tts = manager.load("xtts_v2", _load_xtts)
    tts.tts_to_file(
        text=text,
        speaker_wav=str(ref_path),
        language="en",
        file_path=str(out),
        speed=speed,
        temperature=temperature,
    )


def _f5_generate(text, ref_path, out, steps, speed, ref_text=""):
    tts = manager.load("f5_tts", _load_f5)
    wav, sr, _ = tts.infer(
        ref_file=str(ref_path),
        ref_text=ref_text,
        gen_text=text,
        nfe_step=int(steps),
        speed=speed,
    )
    sf.write(str(out), wav, sr)


def _chatterbox_generate(text, ref_path, out, exaggeration, cfg_weight):
    model = manager.load("chatterbox", _load_chatterbox)
    wav = model.generate(
        text,
        audio_prompt_path=str(ref_path),
        exaggeration=exaggeration,
        cfg_weight=cfg_weight,
    )
    sf.write(str(out), wav.squeeze().cpu().float().numpy(), model.sr)


def generate(
    text: str,
    ref_path: Path,
    progress_ref: list | None = None,
    output_dir: Path | None = None,
    backend: str = "xtts_v2",
    param1: float | None = None,
    param2: float | None = None,
    ref_text: str = "",
) -> Path:
    if output_dir is None:
        output_dir = VC_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = BACKENDS[backend]
    p1 = param1 if param1 is not None else cfg["param1"]["default"]
    p2 = param2 if param2 is not None else cfg["param2"]["default"]

    if progress_ref is not None:
        progress_ref[0] = 0.2

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = output_dir / f"vc_{ts}.wav"

    if progress_ref is not None:
        progress_ref[0] = 0.4

    if backend == "xtts_v2":
        _xtts_generate(text, ref_path, out, temperature=p1, speed=p2)
    elif backend == "f5_tts":
        _f5_generate(text, ref_path, out, steps=p1, speed=p2, ref_text=ref_text)
    elif backend == "chatterbox":
        _chatterbox_generate(text, ref_path, out, exaggeration=p1, cfg_weight=p2)
    else:
        raise ValueError(f"Unknown backend: {backend}")

    if progress_ref is not None:
        progress_ref[0] = 1.0

    return out
