import numpy as np
import soundfile as sf
import torchaudio
from datetime import datetime
from pathlib import Path
from faily.core.model_manager import manager, VC_MODELS_DIR

VC_OUTPUT_DIR = Path("outputs/vc")

_TTS_ID = "microsoft/speecht5_tts"
_VOC_ID = "microsoft/speecht5_hifigan"
_SPK_ID = "speechbrain/spkrec-xvect-voxceleb"


def _load_tts():
    from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
    p   = SpeechT5Processor.from_pretrained(_TTS_ID, cache_dir=str(VC_MODELS_DIR))
    m   = SpeechT5ForTextToSpeech.from_pretrained(_TTS_ID, cache_dir=str(VC_MODELS_DIR)).to(manager.device)
    voc = SpeechT5HifiGan.from_pretrained(_VOC_ID, cache_dir=str(VC_MODELS_DIR)).to(manager.device)
    return p, m, voc


def _load_spk_enc():
    import sys, types
    # k2 (K2-FSA) is an ASR toolkit that SpeechBrain lazily imports via its
    # Xvector module. It has no Windows/Python 3.14 wheels and we don't use it —
    # stub it out so the lazy import doesn't blow up.
    sys.modules.setdefault('k2', types.ModuleType('k2'))
    from speechbrain.inference.classifiers import EncoderClassifier
    from speechbrain.utils.fetching import LocalStrategy
    return EncoderClassifier.from_hparams(
        source=_SPK_ID,
        savedir=str(VC_MODELS_DIR / "spkrec-xvect"),
        local_strategy=LocalStrategy.COPY,
    )


def _speaker_embedding(ref_path: Path):
    import torch
    clf = manager.load(_SPK_ID, _load_spk_enc)

    waveform, sr = torchaudio.load(str(ref_path))
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(0, keepdim=True)

    with torch.no_grad():
        emb = clf.encode_batch(waveform)  # (1, 1, 512)

    return emb.squeeze().unsqueeze(0)  # (1, 512)


def generate(
    text: str,
    ref_path: Path,
    progress_ref: list | None = None,
    output_dir: Path | None = None,
) -> Path:
    import torch
    if output_dir is None:
        output_dir = VC_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    if progress_ref is not None:
        progress_ref[0] = 0.15

    processor, model, vocoder = manager.load(_TTS_ID, _load_tts)

    if progress_ref is not None:
        progress_ref[0] = 0.4

    spk_emb = _speaker_embedding(ref_path).to(manager.device)

    if progress_ref is not None:
        progress_ref[0] = 0.65

    inputs = processor(text=text, return_tensors="pt").to(manager.device)

    with torch.no_grad():
        speech = model.generate_speech(
            inputs["input_ids"],
            speaker_embeddings=spk_emb,
            vocoder=vocoder,
        )

    audio = speech.cpu().numpy()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = output_dir / f"vc_{ts}.wav"
    sf.write(str(out), audio, 16000)

    if progress_ref is not None:
        progress_ref[0] = 1.0

    return out
