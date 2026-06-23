from datetime import datetime
from pathlib import Path
from faily.core.model_manager import manager

VC_OUTPUT_DIR = Path("outputs/vc")
_XTTS_ID = "xtts_v2"


def _load_xtts():
    from TTS.api import TTS
    gpu = str(manager.device).startswith("cuda")
    return TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=gpu)


def generate(
    text: str,
    ref_path: Path,
    progress_ref: list | None = None,
    output_dir: Path | None = None,
    temperature: float = 0.75,
    speed: float = 1.0,
) -> Path:
    if output_dir is None:
        output_dir = VC_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    if progress_ref is not None:
        progress_ref[0] = 0.2

    tts = manager.load(_XTTS_ID, _load_xtts)

    if progress_ref is not None:
        progress_ref[0] = 0.6

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = output_dir / f"vc_{ts}.wav"

    tts.tts_to_file(
        text=text,
        speaker_wav=str(ref_path),
        language="en",
        file_path=str(out),
        speed=speed,
        temperature=temperature,
    )

    if progress_ref is not None:
        progress_ref[0] = 1.0

    return out
