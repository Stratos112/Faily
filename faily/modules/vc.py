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
    # transformers ASR pipeline calls ffmpeg_read(bpayload: bytes, sampling_rate) to decode
    # audio; replace it with a soundfile implementation so FFmpeg doesn't need to be installed.
    import transformers.pipelines.audio_utils as au
    if getattr(au, "_faily_patched", False):
        return
    import io
    import torch
    def _sf_read(bpayload, sampling_rate):
        data, sr = sf.read(io.BytesIO(bpayload), dtype="float32", always_2d=False)
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
        "param2": {"label": "SPEED", "tooltip": "Speech rate. 1.0 is natural pace.", "min": 0.05, "max": 2.0, "step": 0.05, "default": 1.0},
    },
    "f5_tts": {
        "label": "F5-TTS",
        "desc": "SWC Lab · Flow-matching diffusion · Quality scales with steps",
        "param1": {"label": "STEPS", "tooltip": "Diffusion steps. More = higher quality but slower. 32 is a good balance.", "min": 8, "max": 64, "step": 4, "default": 32},
        "param2": {"label": "SPEED", "tooltip": "Speech rate. 1.0 is natural pace.", "min": 0.05, "max": 2.0, "step": 0.05, "default": 1.0},
    },
    "chatterbox": {
        "label": "Chatterbox",
        "desc": "Resemble AI · CFG-guided · Emotion exaggeration control",
        "param1": {"label": "EXAGGERATION", "tooltip": "Emotional intensity. Low = calm and neutral. High = expressive.", "min": 0.0, "max": 1.0, "step": 0.05, "default": 0.5},
        "param2": {"label": "CFG WEIGHT", "tooltip": "Guidance strength. Higher = more faithful to the reference voice style.", "min": 0.0, "max": 1.0, "step": 0.05, "default": 0.5},
    },
    "kokoro": {
        "label": "Kokoro + FreeVC",
        "desc": "Hexgrad Kokoro generates expressive speech → FreeVC applies the character's voice on top. Voice name in VOICE NAME field sets the expression style.",
        "param1": {"label": "SPEED", "tooltip": "Speech rate for the Kokoro expression pass. 1.0 is natural pace.", "min": 0.05, "max": 2.0, "step": 0.05, "default": 1.0},
        "param2": {"label": "LANGUAGE", "tooltip": "0 = American English  1 = British English  2 = Japanese  3 = Mandarin Chinese", "min": 0, "max": 3, "step": 1, "default": 0},
    },
    # styletss2 omitted — dep conflicts with current stack (accelerate<0.26, huggingface-hub<0.20).
}

# Expression engines for the TUNE tab stage-1 pass (text description → expressive audio).
# Stage 2 is always FreeVC (see _freevc_convert).
EXPRESSION_ENGINES = {
    "parler": {
        "label": "Parler-TTS",
        "desc": "Hugging Face · free-text style descriptions → expressive intermediate audio",
    },
    # Future — uncomment when CosyVoice 2 loader is implemented:
    # "cosyvoice2": {
    #     "label": "CosyVoice 2",
    #     "desc": "Alibaba · natural-language style instructions + zero-shot cloning in one pass",
    # },
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


_PARLER_ID = "parler-tts/parler-tts-mini-v1.1"


def _load_parler():
    import sys, torch, inspect
    import transformers.cache_utils as _cu
    import transformers.pytorch_utils as _pu

    # 1. SlidingWindowCache removed in transformers 5.x
    if not hasattr(_cu, "SlidingWindowCache"):
        _cu.SlidingWindowCache = getattr(_cu, "SlidingWindowStaticCache", _cu.StaticCache)
    # 2. isin_mps_friendly removed in transformers 5.x
    if not hasattr(_pu, "isin_mps_friendly"):
        _pu.isin_mps_friendly = torch.isin
    # 3. Drop any partial failed imports so patched versions are picked up cleanly
    for _k in list(sys.modules):
        if _k.startswith("parler_tts"):
            del sys.modules[_k]

    from parler_tts import ParlerTTSForConditionalGeneration
    from parler_tts.configuration_parler_tts import ParlerTTSConfig
    from parler_tts.modeling_parler_tts import ParlerTTSForCausalLM
    from transformers import AutoTokenizer, GenerationMixin

    # 4. to_diff_dict() calls ParlerTTSConfig() with no args → ValueError; block it
    ParlerTTSConfig.has_no_defaults_at_init = True
    # 5. transformers 5.x raises AttributeError for missing config attrs instead of None
    ParlerTTSConfig.tie_encoder_decoder = False

    # 6 & 7. GenerationMixin removed from PreTrainedModel MRO in 5.x; add back to both classes
    if not issubclass(ParlerTTSForConditionalGeneration, GenerationMixin):
        ParlerTTSForConditionalGeneration.__bases__ = (GenerationMixin,) + ParlerTTSForConditionalGeneration.__bases__
    if not issubclass(ParlerTTSForCausalLM, GenerationMixin):
        ParlerTTSForCausalLM.__bases__ = (GenerationMixin,) + ParlerTTSForCausalLM.__bases__

    # 8. _prepare_attention_mask_for_generation: signature changed from
    #    (inputs, pad_tensor, eos_tensor) to (inputs, generation_config, model_kwargs)
    _t5_prep_attn = GenerationMixin._prepare_attention_mask_for_generation
    def _prep_attn_compat(self, inputs_tensor, pad_or_config, eos_or_kwargs=None, model_kwargs=None):
        if isinstance(pad_or_config, torch.Tensor):
            class _Cfg:
                _pad_token_tensor = pad_or_config
                _eos_token_tensor = eos_or_kwargs
            return _t5_prep_attn(self, inputs_tensor, _Cfg(), model_kwargs or {})
        return _t5_prep_attn(self, inputs_tensor, pad_or_config, eos_or_kwargs or {})
    ParlerTTSForConditionalGeneration._prepare_attention_mask_for_generation = _prep_attn_compat

    # 9. tie_weights: 5.x passes extra kwargs parler-tts's override doesn't accept;
    #    also explicitly call text_encoder.tie_weights() to fix T5 embed_tokens meta tensor
    _orig_tie = ParlerTTSForConditionalGeneration.tie_weights
    _orig_tie_params = set(inspect.signature(_orig_tie).parameters) - {"self"}
    def _tie_weights(self, **kwargs):
        result = _orig_tie(self, **{k: v for k, v in kwargs.items() if k in _orig_tie_params})
        if hasattr(self, "text_encoder") and hasattr(self.text_encoder, "tie_weights"):
            try:
                self.text_encoder.tie_weights()
            except Exception:
                pass
        return result
    ParlerTTSForConditionalGeneration.tie_weights = _tie_weights

    # 10. _expand_inputs_for_generation: two issues in 5.x:
    #   a) parler-tts computes expand_size from generation config fields that may be None → default to 1
    #   b) 5.x reordered the signature (is_encoder_decoder moved to position 1 before expand_size),
    #      so passing (input_ids, 1, **kwargs_containing_is_encoder_decoder) duplicates it.
    #      Fix: extract is_encoder_decoder as a named param and use all-keyword call.
    _orig_expand = GenerationMixin._expand_inputs_for_generation
    @staticmethod
    def _expand_compat(input_ids=None, expand_size=1, is_encoder_decoder=False, **kwargs):
        return _orig_expand(
            input_ids=input_ids,
            expand_size=1 if expand_size is None else expand_size,
            is_encoder_decoder=is_encoder_decoder,
            **kwargs,
        )
    ParlerTTSForConditionalGeneration._expand_inputs_for_generation = _expand_compat
    ParlerTTSForCausalLM._expand_inputs_for_generation = _expand_compat

    # 11. _get_initial_cache_position: 5.x added a `device` arg
    #     4.46 call: (self, seq_len, model_kwargs)
    #     5.x call:  (self, seq_len, device, model_kwargs)  ← extra positional arg
    _orig_cache_pos = ParlerTTSForConditionalGeneration._get_initial_cache_position
    def _cache_pos_compat(self, seq_len, device_or_kwargs, model_kwargs=None):
        if model_kwargs is None:
            # Old 3-arg style — forward as-is
            return _orig_cache_pos(self, seq_len, device_or_kwargs)
        # New 4-arg style — drop device, pass seq_len + model_kwargs
        return _orig_cache_pos(self, seq_len, model_kwargs)
    ParlerTTSForConditionalGeneration._get_initial_cache_position = _cache_pos_compat

    # 12. device_map avoids meta-tensor crash on .to(device); float16 for VRAM efficiency
    model = ParlerTTSForConditionalGeneration.from_pretrained(
        _PARLER_ID,
        cache_dir=str(VC_MODELS_DIR),
        device_map={"": manager.device},
        torch_dtype=torch.float16,
    )
    tok = AutoTokenizer.from_pretrained(_PARLER_ID, cache_dir=str(VC_MODELS_DIR))
    return model, tok


def _parler_generate(text: str, out: Path, style_prompt: str):
    import torch
    model, tokenizer = manager.load(_PARLER_ID, _load_parler)
    description = style_prompt.strip() or "A clear, neutral voice at a moderate pace."
    input_ids = tokenizer(description, return_tensors="pt").input_ids.to(manager.device)
    prompt_input_ids = tokenizer(text, return_tensors="pt").input_ids.to(manager.device)
    with torch.no_grad():
        gen = model.generate(
            input_ids=input_ids,
            prompt_input_ids=prompt_input_ids,
        )
    sf.write(str(out), gen.cpu().numpy().squeeze(), model.config.sampling_rate)


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


_KOKORO_LANGS = ["a", "b", "j", "z"]


def _load_freevc():
    _patch_xtts_transformers()
    from TTS.api import TTS
    gpu = str(manager.device).startswith("cuda")
    return TTS("voice_conversion_models/multilingual/vctk/freevc24", gpu=gpu)


def _freevc_convert(source_wav: Path, target_wav: Path, out: Path):
    model = manager.load("freevc24", _load_freevc)
    model.voice_conversion_to_file(
        source_wav=str(source_wav),
        target_wav=str(target_wav),
        file_path=str(out),
    )


def _kokoro_generate(
    text: str,
    out: Path,
    style_prompt: str,
    speed: float,
    lang_idx: int,
    ref_path: Path | None = None,
):
    import numpy as np
    lang = _KOKORO_LANGS[max(0, min(int(lang_idx), len(_KOKORO_LANGS) - 1))]

    def _load():
        from kokoro import KPipeline
        return KPipeline(lang_code=lang)

    pipeline = manager.load(f"kokoro_{lang}", _load)
    voice = style_prompt.strip() or "af_heart"
    chunks = []
    for _, _, audio in pipeline(text, voice=voice, speed=speed):
        chunks.append(audio)
    if not chunks:
        raise RuntimeError("Kokoro produced no audio output")
    audio = np.concatenate(chunks)

    if ref_path is not None and ref_path.exists():
        # Two-stage: Kokoro expression pass → FreeVC character voice conversion
        tmp = out.with_name(f"_tmp_{out.name}")
        sf.write(str(tmp), audio, 24000)
        try:
            _freevc_convert(tmp, ref_path, out)
        finally:
            if tmp.exists():
                tmp.unlink()
    else:
        sf.write(str(out), audio, 24000)


def tune_generate(
    text: str,
    expression: str,
    engine: str,
    ref_path: Path,
    progress_ref: list | None = None,
    output_dir: Path | None = None,
) -> Path:
    """Two-stage TUNE pipeline: expression engine → FreeVC character voice conversion."""
    if output_dir is None:
        output_dir = VC_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = output_dir / f"tune_{ts}.wav"
    stage1 = output_dir / f"_s1_{ts}.wav"

    if progress_ref is not None:
        progress_ref[0] = 0.1

    # Stage 1: expression engine generates styled intermediate audio
    if engine == "parler":
        _parler_generate(text, stage1, style_prompt=expression)
    # elif engine == "cosyvoice2":
    #     _cosyvoice2_generate(text, stage1, style_prompt=expression)
    else:
        raise ValueError(f"Unknown expression engine: {engine!r}")

    if progress_ref is not None:
        progress_ref[0] = 0.6

    # Stage 2: FreeVC converts intermediate to character's voice
    try:
        _freevc_convert(stage1, ref_path, out)
    finally:
        if stage1.exists():
            stage1.unlink()

    if progress_ref is not None:
        progress_ref[0] = 1.0

    return out


def generate(
    text: str,
    ref_path: Path | None,
    progress_ref: list | None = None,
    output_dir: Path | None = None,
    backend: str = "xtts_v2",
    param1: float | None = None,
    param2: float | None = None,
    ref_text: str = "",
    style_prompt: str = "",
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
    elif backend == "kokoro":
        _kokoro_generate(text, out, style_prompt=style_prompt, speed=p1, lang_idx=int(p2), ref_path=ref_path)
    else:
        raise ValueError(f"Unknown backend: {backend}")

    if progress_ref is not None:
        progress_ref[0] = 1.0

    return out
