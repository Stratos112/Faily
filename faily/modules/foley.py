import numpy as np
import soundfile as sf
from datetime import datetime
from pathlib import Path
from faily.core.model_manager import manager, SFX_MODELS_DIR

_BUNDLED: dict[str, str] = {
    "AudioLDM2 (diffusion, versatile)": "cvssp/audioldm2",
    "AudioLDM2 Large":                  "cvssp/audioldm2-large",
}

SFX_OUTPUT_DIR = Path("outputs/sfx")


def scan_local() -> dict[str, str]:
    found: dict[str, str] = {}
    if not SFX_MODELS_DIR.exists():
        return found
    for d in SFX_MODELS_DIR.iterdir():
        if d.is_dir() and not d.name.startswith("models--") and (d / "config.json").exists():
            found[f"[local] {d.name}"] = str(d.resolve())
    return found


def get_models() -> dict[str, str]:
    return {**_BUNDLED, **scan_local()}


def _loader_audioldm2(model_id: str):
    from diffusers import AudioLDM2Pipeline
    from transformers import GPT2LMHeadModel
    import torch
    dtype = torch.float16 if manager.device == "cuda" else torch.float32
    pipe = AudioLDM2Pipeline.from_pretrained(
        model_id, torch_dtype=dtype, cache_dir=str(SFX_MODELS_DIR)
    )
    # cvssp/audioldm2 saves language_model as GPT2Model which lacks GenerationMixin
    pipe.language_model = GPT2LMHeadModel.from_pretrained(
        model_id, subfolder="language_model", torch_dtype=dtype,
        ignore_mismatched_sizes=True, cache_dir=str(SFX_MODELS_DIR),
    )
    return pipe.to(manager.device)


def generate(
    prompt: str,
    model_id: str,
    duration: float = 5.0,
    steps: int = 50,
    guidance: float = 3.5,
    progress_ref: list | None = None,
    output_dir: Path | None = None,
) -> Path:
    if output_dir is None:
        output_dir = SFX_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    pipe = manager.load(model_id, lambda: _loader_audioldm2(model_id))

    def _cb(step: int, timestep: int, latents):
        if progress_ref is not None:
            progress_ref[0] = (step + 1) / steps

    result = pipe(
        prompt,
        num_inference_steps=steps,
        audio_length_in_s=duration,
        guidance_scale=guidance,
        callback=_cb,
        callback_steps=1,
    )

    audio: np.ndarray = result.audios[0]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = output_dir / f"sfx_{ts}.wav"
    sf.write(str(out), audio, 16000)
    return out
