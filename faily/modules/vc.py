import soundfile as sf
import torchaudio
from datetime import datetime
from pathlib import Path
from faily.core.model_manager import manager, VC_MODELS_DIR


def _patch_torchaudio():
    # torchcodec DLLs are missing on this system; replace torchaudio.load globally
    # with a soundfile implementation so any library calling it also works.
    if getattr(torchaudio, "_faily_patched", False):
        return
    import torch
    _orig = torchaudio.load  # keep reference in case anything needs it
    def _sf_load(uri, *args, **kwargs):
        data, sr = sf.read(str(uri), dtype="float32", always_2d=True)
        return torch.from_numpy(data.T).contiguous(), sr
    torchaudio.load = _sf_load
    torchaudio._faily_patched = True

_patch_torchaudio()


def _patch_ffmpeg_read():
    # transformers ASR pipeline calls ffmpeg_read when given a filename path;
    # patch it to use soundfile so FFmpeg doesn't need to be installed.
    import transformers.pipelines.audio_utils as au
    if getattr(au, "_faily_patched", False):
        return
    def _sf_read(filename, sampling_rate):
        import torch
        data, sr = sf.read(str(filename), dtype="float32", always_2d=False)
        if sr != sampling_rate:
            wav = torch.from_numpy(data).unsqueeze(0)
            data = torchaudio.functional.resample(wav, sr, sampling_rate).squeeze(0).numpy()
        return data
    au.ffmpeg_read = _sf_read
    au._faily_patched = True

_patch_ffmpeg_read()

VC_OUTPUT_DIR = Path("outputs/vc")

_TTS_ID = "microsoft/speecht5_tts"
_VOC_ID = "microsoft/speecht5_hifigan"
_SPK_ID = "speechbrain/spkrec-xvect-voxceleb"

BACKENDS = {
    "speecht5": {
        "label": "SpeechT5",
        "desc": "Microsoft · X-Vector speaker embedding · HiFi-GAN vocoder",
        "param1": {"label": "VOICE STRENGTH", "tooltip": "Scales the speaker embedding. Below 1.0 is more neutral, above 1.0 exaggerates the voice's character.", "min": 0.5, "max": 2.0, "step": 0.05, "default": 1.0},
        "param2": {"label": "THRESHOLD", "tooltip": "Mel spectrogram stopping criterion. Lower = crisper and shorter output. Higher = smoother but may trail off.", "min": 0.1, "max": 0.9, "step": 0.05, "default": 0.5},
    },
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


def _load_tts():
    from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
    p   = SpeechT5Processor.from_pretrained(_TTS_ID, cache_dir=str(VC_MODELS_DIR))
    m   = SpeechT5ForTextToSpeech.from_pretrained(_TTS_ID, cache_dir=str(VC_MODELS_DIR)).to(manager.device)
    voc = SpeechT5HifiGan.from_pretrained(_VOC_ID, cache_dir=str(VC_MODELS_DIR)).to(manager.device)
    return p, m, voc


def _patch_speechbrain_lazy_modules():
    from speechbrain.utils import importutils
    _orig = importutils.LazyModule.__getattr__
    def _safe(self, attr):
        if attr.startswith("__"):
            raise AttributeError(attr)
        return _orig(self, attr)
    importutils.LazyModule.__getattr__ = _safe


def _load_spk_enc():
    _patch_speechbrain_lazy_modules()
    from speechbrain.inference.classifiers import EncoderClassifier
    from speechbrain.utils.fetching import LocalStrategy
    return EncoderClassifier.from_hparams(
        source=_SPK_ID,
        savedir=str(VC_MODELS_DIR / "spkrec-xvect"),
        local_strategy=LocalStrategy.COPY,
        run_opts={"device": "cpu"},
    )


def _speaker_embedding(ref_path: Path):
    import torch
    clf = manager.load(_SPK_ID, _load_spk_enc)
    data, sr = sf.read(str(ref_path), dtype="float32", always_2d=True)
    waveform = torch.from_numpy(data.T)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(0, keepdim=True)
    with torch.no_grad():
        emb = clf.encode_batch(waveform)
    return emb.squeeze().unsqueeze(0)


def _speecht5_generate(text, ref_path, out, emb_scale, threshold):
    import torch
    processor, model, vocoder = manager.load(_TTS_ID, _load_tts)
    spk_emb = _speaker_embedding(ref_path).to(manager.device) * emb_scale
    inputs = processor(text=text, return_tensors="pt").to(manager.device)
    with torch.no_grad():
        speech = model.generate_speech(
            inputs["input_ids"],
            speaker_embeddings=spk_emb,
            vocoder=vocoder,
            threshold=threshold,
        )
    sf.write(str(out), speech.cpu().numpy(), 16000)


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


def transcribe_ref(ref_path: Path) -> str:
    from f5_tts.infer.utils_infer import transcribe
    return transcribe(str(ref_path))


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

    if backend == "speecht5":
        _speecht5_generate(text, ref_path, out, emb_scale=p1, threshold=p2)
    elif backend == "xtts_v2":
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
