import numpy as np
import soundfile as sf
from datetime import datetime
from pathlib import Path
from faily.core.model_manager import manager, TTS_MODELS_DIR

_BUNDLED: dict[str, str] = {
    "MMS-TTS (fast, CPU-friendly)": "facebook/mms-tts-eng",
    "Bark Small (expressive)":      "suno/bark-small",
    "Bark (high quality)":          "suno/bark",
}

TTS_OUTPUT_DIR = Path("outputs/tts")


def scan_local() -> dict[str, str]:
    found: dict[str, str] = {}
    if not TTS_MODELS_DIR.exists():
        return found
    for d in TTS_MODELS_DIR.iterdir():
        if d.is_dir() and not d.name.startswith("models--") and (d / "config.json").exists():
            found[f"[local] {d.name}"] = str(d.resolve())
    return found


def get_models() -> dict[str, str]:
    return {**_BUNDLED, **scan_local()}


def _loader(model_id: str):
    from transformers import pipeline
    return pipeline(
        "text-to-speech", model=model_id, device=manager.device,
        model_kwargs={"cache_dir": str(TTS_MODELS_DIR)},
    )


def generate(
    text: str,
    model_id: str,
    speed: float = 1.0,
    progress_ref: list | None = None,
    output_dir: Path | None = None,
) -> Path:
    if output_dir is None:
        output_dir = TTS_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    pipe = manager.load(model_id, lambda: _loader(model_id))

    # progress > 0 signals the UI to switch from loader animation to progress bar
    if progress_ref is not None:
        progress_ref[0] = 0.3

    result = pipe(text)

    audio: np.ndarray = np.array(result["audio"]).squeeze()
    rate: int = result["sampling_rate"]

    if speed != 1.0:
        import scipy.signal
        audio = scipy.signal.resample(audio, int(len(audio) / speed)).astype(np.float32)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = output_dir / f"tts_{ts}.wav"
    sf.write(str(out), audio, rate)

    if progress_ref is not None:
        progress_ref[0] = 1.0

    return out
