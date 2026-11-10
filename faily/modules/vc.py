import re
import time
import shutil
import tempfile
import soundfile as sf
import torchaudio
from pathlib import Path
from faily.core.model_manager import manager, VC_MODELS_DIR
from faily.core.paths import history_dir
from faily.modules.edit import ensure_stereo


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


def _make_clip_name(char_name: str | None, style: str, text: str, output_dir: Path) -> str:
    """Build a human-readable clip filename: {char}_{style_word}_{text_word}_{NNN}.wav"""
    def _first_word(s: str) -> str:
        m = re.search(r'[a-z0-9]+', s.lower())
        return m.group(0)[:12] if m else ""

    parts = []
    if char_name:
        slug = re.sub(r'[^a-z0-9]+', '_', char_name.lower().strip()).strip('_')[:16]
        if slug:
            parts.append(slug)
    if style:
        w = _first_word(style)
        if w:
            parts.append(w)
    if text:
        w = _first_word(text)
        if w:
            parts.append(w)
    if not parts:
        parts = ["clip"]

    base = "_".join(parts)
    existing = {p.stem for p in output_dir.glob("*.wav")} if output_dir.exists() else set()
    for i in range(1, 10000):
        stem = f"{base}_{i:03d}"
        if stem not in existing:
            return f"{stem}.wav"
    return f"{base}_9999.wav"

BACKENDS = {
    "xtts_v2": {
        "label": "XTTS v2",
        "desc": "Coqui AI · Zero-shot cross-attention conditioning. Best all-rounder for voice cloning. Natural prosody on short-to-medium clips. Slow initial load; can clip or cut off on very long inputs.",
        "text_hint": "Ideal: 1–2 sentences, under ~250 characters per call. Long unbroken text is where it clips or cuts off — split longer lines into multiple generations.",
        "split_chars": 250,
        "param1": {"label": "TEMPERATURE", "tooltip": "Expressiveness. Low = flat and consistent. High = emotive but may wander.", "min": 0.1, "max": 1.0, "step": 0.05, "default": 0.75},
        "param2": {"label": "SPEED", "tooltip": "Speech rate. 1.0 is natural pace.", "min": 0.05, "max": 2.0, "step": 0.05, "default": 1.0},
    },
    "f5_tts": {
        "label": "F5-TTS",
        "desc": "SWC Lab · Flow-matching diffusion. Highest speaker similarity when given a good transcript. Requires the reference transcript — auto-fills on upload. More steps = better quality but slower. Overkill for quick drafts.",
        "text_hint": "Ideal: keep generated text roughly proportional to reference length (up to ~2–3× the ref duration). Output duration is predicted from that ratio, so a long line against a short ref causes pacing drift.",
        "split_chars": 300,
        "param1": {"label": "STEPS", "tooltip": "Diffusion steps. More = higher quality but slower. 32 is a good balance.", "min": 8, "max": 64, "step": 4, "default": 32},
        "param2": {"label": "SPEED", "tooltip": "Speech rate. 1.0 is natural pace.", "min": 0.05, "max": 2.0, "step": 0.05, "default": 1.0},
    },
    "chatterbox": {
        "label": "Chatterbox",
        "desc": "Resemble AI · CFG-guided generation. Strong emotional range via the exaggeration dial. Less faithful to exact speaker identity than XTTS. Good for expressive or theatrical characters.",
        "text_hint": "Ideal: a few sentences, up to ~30–40s of output. Beyond that, exaggeration and pacing start to wander.",
        "split_chars": 450,
        "param1": {"label": "EXAGGERATION", "tooltip": "Emotional intensity. Low = calm and neutral. High = expressive.", "min": 0.0, "max": 1.0, "step": 0.05, "default": 0.5},
        "param2": {"label": "CFG WEIGHT", "tooltip": "Guidance strength. Higher = more faithful to the reference voice style.", "min": 0.0, "max": 1.0, "step": 0.05, "default": 0.5},
    },
    # styletss2 omitted — dep conflicts with current stack (accelerate<0.26, huggingface-hub<0.20).
}

STAGE2_BACKENDS = {
    "freevc": {
        "label": "FreeVC",
        "desc": "Fast zero-shot voice swap. Reliable on short inputs under ~4s. Quality degrades noticeably on longer Parler output — reduce MAX TOKENS if output sounds garbled.",
    },
    "openvoice": {
        "label": "OpenVoice v2",
        "desc": "MyShell AI · zero-shot tone color conversion. Best results when paired with MeloTTS as the expression engine — they share the same base speaker model and the voice transfer is exact. Parler/Kokoro as stage 1 may produce non-English prosody artefacts.",
    },
    "seedvc": {
        "label": "Seed-VC",
        "desc": "Bytedance · diffusion-based zero-shot VC. Uses Whisper for speaker-agnostic content features — prosody and rhythm from the expression engine transfer cleanly regardless of source voice. Slower than FreeVC; 10 diffusion steps is a good quality/speed balance.",
    },
}

# Expression engines for the TUNE tab stage-1 pass (text description → expressive audio).
EXPRESSION_ENGINES = {
    "parler": {
        "label": "Parler-TTS",
        "desc": "Parler-TTS · Describe tone, pace, and delivery in plain English. Expressive but inconsistent — results vary by phrasing. Lower MAX TOKENS reduces downstream garbling in voice conversion.",
    },
    # Future — uncomment when CosyVoice 2 loader is implemented:
    # "cosyvoice2": {
    #     "label": "CosyVoice 2",
    #     "desc": "Alibaba · natural-language style instructions + zero-shot cloning in one pass",
    # },
}


def _patch_xtts_transformers():
    # transformers 5.x removed isin_mps_friendly; coqui-tts still imports it
    import torch
    import transformers.pytorch_utils as pu
    if not hasattr(pu, "isin_mps_friendly"):
        pu.isin_mps_friendly = torch.isin


def _coqui_env():
    # Coqui asks for CPML acceptance via input() on first model download — a
    # headless server has no stdin, so that blocks/crashes. Accepted once
    # interactively on the old Windows setup; keep accepting it here.
    # TTS_HOME puts coqui's model cache under models/ (a persisted volume)
    # instead of ~/.local/share/tts, which is lost on every image rebuild.
    import os
    os.environ.setdefault("COQUI_TOS_AGREED", "1")
    os.environ.setdefault("TTS_HOME", str((VC_MODELS_DIR / "coqui").resolve()))


def _load_xtts():
    _patch_xtts_transformers()
    _coqui_env()
    from TTS.api import TTS
    gpu = str(manager.device).startswith("cuda")
    return TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=gpu)


def transcribe_ref(ref_path: Path) -> str:
    _patch_ffmpeg_read()
    from f5_tts.infer.utils_infer import transcribe
    # Hand the ASR pipeline decoded samples, not a path: transformers 5 decodes
    # paths through torchcodec (bypassing _patch_ffmpeg_read entirely), and
    # torchcodec's native libs are the fragile part of this stack.
    data, sr = sf.read(str(ref_path), dtype="float32", always_2d=True)
    return transcribe({"raw": data.mean(axis=1), "sampling_rate": sr})


def _load_f5():
    from f5_tts.api import F5TTS
    return F5TTS(device=str(manager.device))


def _load_chatterbox():
    from chatterbox.tts import ChatterboxTTS
    return ChatterboxTTS.from_pretrained(device=str(manager.device))


_PARLER_ID = "parler-tts/parler-tts-mini-v1.1"


def _patch_parler_cache():
    """parler-tts builds its KV cache with EncoderDecoderCache.from_legacy_cache, which
    transformers 5 removed, so use_cache=True crashed and we ran without a cache (~1.7x
    slower, same quality). Provide the two legacy-cache helpers it relies on."""
    import transformers.cache_utils as cu
    if not hasattr(cu.EncoderDecoderCache, "from_legacy_cache"):
        def _from_legacy(cls, past=None):
            if past is None:
                return cls(cu.DynamicCache(), cu.DynamicCache())
            self_attn, cross_attn = cu.DynamicCache(), cu.DynamicCache()
            for i, layer in enumerate(past):
                self_attn.update(layer[0], layer[1], i)
                if len(layer) > 2:
                    cross_attn.update(layer[2], layer[3], i)
            return cls(self_attn, cross_attn)
        cu.EncoderDecoderCache.from_legacy_cache = classmethod(_from_legacy)
    if not hasattr(cu.EncoderDecoderCache, "to_legacy_cache"):
        cu.EncoderDecoderCache.to_legacy_cache = lambda self: self


def _load_parler():
    import sys, torch, inspect
    import transformers.cache_utils as _cu
    import transformers.pytorch_utils as _pu

    _patch_parler_cache()
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
    from parler_tts.modeling_parler_tts import ParlerTTSForCausalLM, ParlerTTSDecoder
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

    # 12. _update_causal_mask: 5.x outer generate loop sets cache_position to the global
    #     sequence positions (e.g. [0..6] for a 7-token input), but parler-tts's internal
    #     decoder calls this with input_tensor whose seq dim may be smaller (e.g. 2 codec
    #     positions per chunk).  The multiplication requires both to have the same dim-0.
    #     Fix: if lengths differ, trim cache_position to seq_len from the front.
    _orig_ucm = ParlerTTSDecoder._update_causal_mask
    def _ucm_compat(self, attention_mask, input_tensor, cache_position, past_key_values, output_attentions):
        if cache_position is not None:
            seq_len = input_tensor.shape[1]
            if cache_position.shape[0] > seq_len:
                cache_position = cache_position[:seq_len]
            elif cache_position.shape[0] < seq_len:
                cache_position = torch.arange(seq_len, device=input_tensor.device)
        return _orig_ucm(self, attention_mask, input_tensor, cache_position, past_key_values, output_attentions)
    ParlerTTSDecoder._update_causal_mask = _ucm_compat

    # 13. DynamicCache.key_cache / value_cache were removed in transformers 5.x.
    #     Parler-tts reads them directly as lists indexed by layer_idx.
    #     Patch update() to maintain parallel lists so parler-tts's reads work.
    if not hasattr(_cu.DynamicCache, 'key_cache'):
        _orig_dc_update = _cu.DynamicCache.update
        def _dc_update_compat(self, key_states, value_states, layer_idx, cache_kwargs=None, **kwargs):
            key_full, value_full = _orig_dc_update(self, key_states, value_states, layer_idx, cache_kwargs, **kwargs)
            if 'key_cache' not in self.__dict__:
                self.__dict__['key_cache'] = []
                self.__dict__['value_cache'] = []
            kc = self.__dict__['key_cache']
            vc = self.__dict__['value_cache']
            while len(kc) <= layer_idx:
                kc.append(None)
                vc.append(None)
            kc[layer_idx] = key_full
            vc[layer_idx] = value_full
            return key_full, value_full
        _cu.DynamicCache.update = _dc_update_compat

        _orig_get_seq_len = _cu.DynamicCache.get_seq_length
        def _get_seq_len_compat(self, layer_idx=0):
            kc = self.__dict__.get('key_cache')
            if kc:
                first = next((k for k in kc if k is not None), None)
                if first is not None:
                    return first.shape[-2]
            return _orig_get_seq_len(self, layer_idx)
        _cu.DynamicCache.get_seq_length = _get_seq_len_compat

    # 14. prepare_inputs_for_generation: in 5.x, an empty DynamicCache is pre-allocated
    #     before the prefill and passed as past_key_values (get_seq_length()=0). Parler-tts
    #     was written for 4.46 where past_key_values=None at the start, so the formula
    #     generated_length = 0 - prompt_len + N goes negative → crash. Fix: treat any
    #     empty cache as None so parler-tts takes its else branch (generated_length = N).
    _orig_pig = ParlerTTSForConditionalGeneration.prepare_inputs_for_generation
    def _pig_compat(self, decoder_input_ids, past_key_values=None, **kwargs):
        if (past_key_values is not None and
                hasattr(past_key_values, 'get_seq_length') and
                past_key_values.get_seq_length() == 0):
            past_key_values = None
        return _orig_pig(self, decoder_input_ids, past_key_values=past_key_values, **kwargs)
    ParlerTTSForConditionalGeneration.prepare_inputs_for_generation = _pig_compat

    # 16. device_map avoids meta-tensor crash on .to(device); float16 for VRAM efficiency
    from faily.core.hf_token import get_hf_token
    token = get_hf_token()
    model = ParlerTTSForConditionalGeneration.from_pretrained(
        _PARLER_ID,
        cache_dir=str(VC_MODELS_DIR),
        device_map={"": manager.device},
        torch_dtype=torch.float16,
        token=token,
    )
    tok = AutoTokenizer.from_pretrained(_PARLER_ID, cache_dir=str(VC_MODELS_DIR), token=token)
    return model, tok


def _normalize_audio(path: Path, target_db: float):
    import numpy as np
    data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    rms = np.sqrt(np.mean(data ** 2))
    if rms < 1e-9:
        return
    target_rms = 10 ** (target_db / 20)
    data = np.clip(data * (target_rms / rms), -1.0, 1.0)
    sf.write(str(path), data, sr)


def _parler_generate(text: str, out: Path, style_prompt: str, max_new_tokens: int = 500):
    import torch
    model, tokenizer = manager.load(_PARLER_ID, _load_parler)
    description = style_prompt.strip() or "A clear, neutral voice at a moderate pace."
    input_ids = tokenizer(description, return_tensors="pt").input_ids.to(manager.device)
    prompt_input_ids = tokenizer(text, return_tensors="pt").input_ids.to(manager.device)
    attention_mask = torch.ones_like(input_ids)
    prompt_attention_mask = torch.ones_like(prompt_input_ids)
    device_type = "cuda" if str(manager.device).startswith("cuda") else "cpu"
    # Parler's codec runs at 86 frames/s, so 500 tokens is only ~5.8 s of speech: scale the
    # budget with the line (~13 chars/s, +40% slack) so longer lines aren't cut off.
    budget = max(max_new_tokens, min(2600, int(len(text) / 13 * 86 * 1.4)))
    expected = len(text) / 15.0                     # seconds of speech this line needs, roughly
    for attempt in range(3):
        with torch.no_grad(), torch.autocast(device_type=device_type, dtype=torch.float16):
            gen = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                prompt_input_ids=prompt_input_ids,
                prompt_attention_mask=prompt_attention_mask,
                max_new_tokens=budget,
                use_cache=True,
                # transformers 4.x silently sampled with top_k=50; 5.x samples the whole code
                # distribution unless told otherwise, which turns the speech to mush (WER ~0.5 -> ~0.1).
                top_k=50,
            )
        wav = gen.cpu().float().numpy().squeeze()
        # sampling occasionally emits end-of-speech almost at once, leaving a fragment of the line
        if len(wav) / model.config.sampling_rate >= 0.5 * expected or attempt == 2:
            break
    sf.write(str(out), wav, model.config.sampling_rate)


PARLER_TRIES = 3
PARLER_MAX_WER = 0.20


def _parler_generate_checked(text: str, out: Path, style_prompt: str, max_new_tokens: int = 500,
                             tries: int = PARLER_TRIES, max_wer: float = PARLER_MAX_WER) -> float:
    """Parler samples, so some takes drop or garble words, and the voice converter then
    faithfully converts the garbage. Generate up to `tries` takes, let Whisper check each
    against the script, stop at the first good one and keep the best otherwise.
    Returns the kept take's word error rate."""
    from faily.modules.asr import transcribe
    from faily.modules.qualify import normalize_words, word_error_rate

    want = normalize_words(text)
    best: Path | None = None
    best_wer = float("inf")
    for attempt in range(tries):
        take = out.with_name(f"_take{attempt}_{out.name}")
        _parler_generate(text, take, style_prompt, max_new_tokens)
        audio, sr = sf.read(str(take), dtype="float32", always_2d=False)
        wer = word_error_rate(want, normalize_words(transcribe(audio.mean(axis=1) if audio.ndim > 1 else audio, sr)))
        if wer < best_wer:
            if best is not None:
                best.unlink(missing_ok=True)
            best, best_wer = take, wer
        else:
            take.unlink(missing_ok=True)
        if best_wer <= max_wer:
            break
    best.replace(out)
    return best_wer


_XTTS_MIN_PIECE = 3.0      # seconds; XTTS wants each reference file > 3 s


def _split_reference(ref_path: Path, piece_seconds: float, min_seconds: float = _XTTS_MIN_PIECE):
    """Cut a long reference into consecutive pieces of `piece_seconds` (a short tail joins the last
    piece). Returns (paths, temp_dir_or_None); a reference that already fits comes back as itself."""
    import numpy as np
    info = sf.info(str(ref_path))
    if info.duration <= piece_seconds + min_seconds:
        return [Path(ref_path)], None
    data, sr = sf.read(str(ref_path), dtype="float32", always_2d=False)
    step = int(piece_seconds * sr)
    starts = list(range(0, len(data), step))
    if len(data) - starts[-1] < int(min_seconds * sr):
        starts.pop()
    bounds = list(zip(starts, starts[1:] + [len(data)]))
    tmp_dir = Path(tempfile.mkdtemp(prefix="xtts_ref_"))
    paths = []
    for i, (a, b) in enumerate(bounds):
        path = tmp_dir / f"piece_{i:03d}.wav"
        sf.write(str(path), np.ascontiguousarray(data[a:b]), sr)
        paths.append(path)
    return paths, tmp_dir


def _xtts_generate(text, ref_path, out, temperature, speed):
    tts = manager.load("xtts_v2", _load_xtts)
    # XTTS keeps only the first max_ref_len seconds of each reference FILE, but averages the speaker
    # embedding over every file it is given: hand it a long reference as pieces of that length.
    cfg = getattr(getattr(getattr(tts, "synthesizer", None), "tts_model", None), "config", None)
    piece = float(getattr(cfg, "max_ref_len", None) or 10)
    pieces, tmp_dir = _split_reference(Path(ref_path), piece)
    try:
        tts.tts_to_file(
            text=text,
            speaker_wav=[str(p) for p in pieces] if len(pieces) > 1 else str(pieces[0]),
            language="en",
            file_path=str(out),
            speed=speed,
            temperature=temperature,
        )
    finally:
        if tmp_dir is not None:
            shutil.rmtree(tmp_dir, ignore_errors=True)


def _f5_short_reference(ref_path: Path, ref_text: str, limit: float = 12.0):
    """F5 itself clips a reference over 12 s but leaves the transcript whole, so the text
    that no longer has audio gets spoken in the output. When a long reference gets through
    anyway, cut it at its quietest moment under the limit and re-transcribe what is left.
    Returns (path, text, temp_path_to_delete_or_None)."""
    import numpy as np
    data, sr = sf.read(str(ref_path), dtype="float32", always_2d=False)
    mono = data.mean(axis=1) if data.ndim > 1 else data
    if len(mono) / sr <= limit:
        return ref_path, ref_text, None
    lo, hi = int((limit - 4.0) * sr), int((limit - 0.5) * sr)
    win = int(0.05 * sr)
    energy = [float(np.mean(mono[i:i + win] ** 2)) for i in range(lo, hi, win)]
    cut = lo + int(np.argmin(energy)) * win
    clipped = mono[:cut]
    from faily.modules.asr import transcribe
    text = transcribe(clipped, sr).strip()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        tmp = Path(f.name)
    sf.write(str(tmp), clipped, sr)
    return tmp, text, tmp


def _f5_generate(text, ref_path, out, steps, speed, ref_text=""):
    ref_path, ref_text, tmp = _f5_short_reference(Path(ref_path), ref_text)
    try:
        _f5_infer(text, ref_path, out, steps, speed, ref_text)
    finally:
        if tmp is not None and tmp.exists():
            tmp.unlink()


def _f5_seed() -> int:
    # F5's infer() writes its seed into os.environ["PYTHONHASHSEED"]; its default draw goes up to
    # sys.maxsize, past CPython's 2**32-1 limit, and every Python subprocess started afterwards
    # (piper_venv: training, diagnose) then dies at startup. Draw one that is valid.
    import random
    return random.randint(0, 2**32 - 1)


def _f5_infer(text, ref_path, out, steps, speed, ref_text=""):
    tts = manager.load("f5_tts", _load_f5)
    wav, sr, _ = tts.infer(
        ref_file=str(ref_path),
        ref_text=ref_text,
        gen_text=text,
        nfe_step=int(steps),
        speed=speed,
        seed=_f5_seed(),
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



_OPENVOICE_CKPT = VC_MODELS_DIR / "openvoice_v2"
_OV_CONV_ID = "openvoice_v2_converter"


def _load_openvoice_converter():
    from openvoice.api import ToneColorConverter
    ckpt_dir = _OPENVOICE_CKPT / "converter"
    config_path = ckpt_dir / "config.json"
    ckpt_path   = ckpt_dir / "checkpoint.pth"
    if not ckpt_path.exists():
        from huggingface_hub import snapshot_download
        from faily.core.hf_token import get_hf_token
        snapshot_download(
            "myshell-ai/openvoice-v2",
            local_dir=str(_OPENVOICE_CKPT),
            ignore_patterns=["*.md", "*.txt"],
            token=get_hf_token(),
        )
    conv = ToneColorConverter(str(config_path), device=str(manager.device))
    conv.load_ckpt(str(ckpt_path))
    return conv


_FREEVC_SPK_URL = "https://github.com/coqui-ai/TTS/releases/download/v0.13.0_models/speaker_encoder.pt"


def _freevc_speaker_encoder() -> str:
    """Local copy of FreeVC's speaker encoder. Coqui hands the GitHub release URL to
    fsspec, whose exists() check fails on GitHub's signed redirect (FileNotFoundError
    even though the file downloads fine), so fetch it ourselves once and load by path."""
    import urllib.request
    dest = VC_MODELS_DIR / "coqui" / "freevc_speaker_encoder.pt"
    if not dest.exists() or dest.stat().st_size < 1_000_000:
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".part")
        for attempt in range(4):
            try:
                urllib.request.urlretrieve(_FREEVC_SPK_URL, tmp)
                break
            except OSError:                       # flaky DNS/network: back off and retry
                if attempt == 3:
                    raise
                time.sleep(2 * (attempt + 1))
        tmp.replace(dest)
    return str(dest.resolve())


def _load_freevc():
    _patch_xtts_transformers()
    _coqui_env()
    from TTS.vc.models import freevc as _fv
    if not getattr(_fv.FreeVC, "_faily_patched", False):
        def _load_spk(self):
            self.enc_spk_ex = _fv.SpeakerEncoderEx(_freevc_speaker_encoder())
        _fv.FreeVC.load_pretrained_speaker_encoder = _load_spk
        _fv.FreeVC._faily_patched = True
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




def tune_generate(
    text: str,
    expression: str,
    engine: str,
    ref_path: Path,
    progress_ref: list | None = None,
    output_dir: Path | None = None,
    char_name: str | None = None,
    normalize_db: float | None = -18.0,
    max_new_tokens: int = 500,
    stage2_backend: str = "freevc",
    ov_tau: float = 0.3,
    svc_steps: int = 10,
    svc_cfg: float = 0.7,
    verify_stage1: bool = True,
) -> Path:
    """Two-stage TUNE pipeline: expression engine → voice conversion.
    verify_stage1: have Whisper check the expression take against the script and redo bad
    ones (up to PARLER_TRIES takes) before converting."""
    if output_dir is None:
        output_dir = history_dir(char_name, ref_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    clip_name = _make_clip_name(char_name, expression, text, output_dir)
    out = output_dir / clip_name
    stage1 = output_dir / f"_s1_{clip_name}"

    if progress_ref is not None:
        progress_ref[0] = 0.1

    # Stage 1: expression engine → expressive intermediate audio
    if engine == "parler":
        if verify_stage1:
            _parler_generate_checked(text, stage1, expression, max_new_tokens)
        else:
            _parler_generate(text, stage1, style_prompt=expression, max_new_tokens=max_new_tokens)
    else:
        raise ValueError(f"Unknown expression engine: {engine!r}")

    if normalize_db is not None:
        _normalize_audio(stage1, normalize_db)

    if progress_ref is not None:
        progress_ref[0] = 0.6

    # Stage 2: voice conversion → character's voice
    try:
        if stage2_backend == "freevc":
            _freevc_convert(stage1, ref_path, out)
        elif stage2_backend == "openvoice":
            _openvoice_convert(stage1, ref_path, out, tau=ov_tau)
        elif stage2_backend == "seedvc":
            _seedvc_convert(stage1, ref_path, out, steps=svc_steps, cfg_rate=svc_cfg)
        else:
            raise ValueError(f"Unknown stage2 backend: {stage2_backend!r}")
    finally:
        if stage1.exists():
            stage1.unlink()

    if progress_ref is not None:
        progress_ref[0] = 1.0

    ensure_stereo(out)
    return out


def _openvoice_convert(source_wav: Path, target_wav: Path, out: Path, tau: float = 0.3):
    import sys, types, os, shutil, tempfile as _tmp
    # whisper_timestamped is imported at the top of se_extractor.py but only
    # used inside split_audio_whisper, which we patch out below.
    for _name, _attrs in [
        ("whisper_timestamped",            []),
        ("whisper_timestamped.transcribe", ["get_audio_tensor", "get_vad_segments"]),
    ]:
        if _name not in sys.modules:
            m = types.ModuleType(_name)
            for a in _attrs:
                setattr(m, a, None)
            sys.modules[_name] = m
    try:
        from openvoice import se_extractor
    except ImportError as _e:
        raise RuntimeError(
            f"OpenVoice dependency missing: {_e}\n"
            'Rebuild the image, or manually:\n'
            '  pip install --no-deps "git+https://github.com/myshell-ai/OpenVoice.git"\n'
            "  pip install wavmark resampy faster-whisper cn2an eng_to_ipa langid jieba"
        ) from _e
    # split_audio_whisper always runs (even with vad=False in this OV version)
    # and requires WhisperModel for VAD segment detection. We don't need that —
    # replace it with a passthrough that treats the whole clip as one segment.
    def _passthrough_split(audio_path, target_dir=None, audio_name=None):
        d = _tmp.mkdtemp()
        shutil.copy2(audio_path, os.path.join(d, "seg_0.wav"))
        return d
    se_extractor.split_audio_whisper = _passthrough_split
    conv = manager.load(_OV_CONV_ID, _load_openvoice_converter)

    # The se_extractor expects mono audio at the model's native sample rate (22050 Hz).
    # Passing multi-channel or wrong-SR audio gives a distorted speaker embedding that
    # can map to unexpected regions of the embedding space (heard as non-English prosody).
    # Write clean mono 22050 Hz temp files for both source and target before extraction.
    import soundfile as _sf
    import numpy as _np

    def _clean_wav(src: Path) -> Path:
        data, sr = _sf.read(str(src), dtype="float32", always_2d=False)
        if data.ndim > 1:
            data = data.mean(axis=1)
        if sr != 22050:
            import librosa as _lr
            data = _lr.resample(data, orig_sr=sr, target_sr=22050)
        tmp_path = Path(_tmp.mktemp(suffix=".wav"))
        _sf.write(str(tmp_path), data, 22050)
        return tmp_path

    tgt_clean = _clean_wav(target_wav)
    src_clean = _clean_wav(source_wav)
    # get_se writes per-call scratch into target_dir (default "processed",
    # relative to cwd) and never cleans it up.
    se_scratch = _tmp.mkdtemp(prefix="faily_ov_se_")
    try:
        src_se, _ = se_extractor.get_se(str(src_clean), conv, target_dir=se_scratch, vad=False)
        tgt_se, _ = se_extractor.get_se(str(tgt_clean), conv, target_dir=se_scratch, vad=False)
    finally:
        for _p in (src_clean, tgt_clean):
            try: _p.unlink()
            except OSError: pass
        shutil.rmtree(se_scratch, ignore_errors=True)
    conv.convert(
        audio_src_path=str(source_wav),
        src_se=src_se,
        tgt_se=tgt_se,
        output_path=str(out),
        tau=tau,
    )


_SEEDVC_DIR = VC_MODELS_DIR / "seed-vc"
_SEEDVC_CACHE = VC_MODELS_DIR / "seed-vc-checkpoints"


def _load_seedvc():
    import sys
    repo = _SEEDVC_DIR.resolve()
    if not repo.exists():
        raise RuntimeError(
            "Seed-VC repo not found. Clone it first:\n"
            f'  git clone https://github.com/Plachtaa/seed-vc.git "{repo}"\n'
            f'  pip install -r "{repo / "requirements.txt"}"'
        )
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    # Seed-VC vendors its own BigVGAN (modules/bigvgan), whose _from_pretrained
    # still requires proxies/resume_download as keyword-only args. Current
    # huggingface_hub no longer passes them → TypeError on load. Default them
    # here rather than editing huggingface_hub itself (which would push the
    # extra kwargs into every PyTorchModelHubMixin model's constructor).
    from modules.bigvgan import bigvgan as _bv
    if not getattr(_bv.BigVGAN, "_faily_patched", False):
        _orig_fp = _bv.BigVGAN._from_pretrained.__func__
        def _fp(cls, *, proxies=None, resume_download=False, **kwargs):
            return _orig_fp(cls, proxies=proxies, resume_download=resume_download, **kwargs)
        _bv.BigVGAN._from_pretrained = classmethod(_fp)
        _bv.BigVGAN._faily_patched = True

    # Seed-VC downloads its weights with cache_dir="./checkpoints", i.e.
    # relative to wherever the app was started — in the container that's
    # /app/checkpoints, outside every volume, so it'd re-download on each
    # container recreate. Redirect into models/ (a volume). Must happen before
    # seed_vc_wrapper is imported: it binds the function by name.
    import hf_utils as _hf
    if not getattr(_hf, "_faily_patched", False):
        from huggingface_hub import hf_hub_download
        def _load_from_hf(repo_id, model_filename="pytorch_model.bin", config_filename=None):
            cache = str(_SEEDVC_CACHE.resolve())
            model_path = hf_hub_download(repo_id=repo_id, filename=model_filename, cache_dir=cache)
            if config_filename is None:
                return model_path
            return model_path, hf_hub_download(repo_id=repo_id, filename=config_filename, cache_dir=cache)
        _hf.load_custom_model_from_hf = _load_from_hf
        _hf._faily_patched = True

    import torch as _torch
    import seed_vc_wrapper as _svc_mod
    return _svc_mod.SeedVCWrapper(device=_torch.device(str(manager.device)))


def _seedvc_convert(source_wav: Path, target_wav: Path, out: Path,
                    steps: int = 10, cfg_rate: float = 0.7):
    import soundfile as _sf
    import numpy as _np

    wrapper = manager.load("seedvc", _load_seedvc)
    gen = wrapper.convert_voice(
        source=str(source_wav),
        target=str(target_wav),
        diffusion_steps=steps,
        length_adjust=1.0,
        inference_cfg_rate=cfg_rate,
        f0_condition=False,
        auto_f0_adjust=True,
        pitch_shift=0,
        stream_output=False,
    )

    # stream_output=False uses `return` (not `yield`) in the generator,
    # so the audio lands in StopIteration.value — capture it manually.
    audio = None
    chunks = []
    try:
        while True:
            val = next(gen)
            if val is not None:
                chunks.append(val)
    except StopIteration as _si:
        if _si.value is not None:
            audio = _si.value

    if audio is None and chunks:
        # fallback: it yielded values directly
        audio = chunks[-1]  # last chunk is the full audio in some versions

    if audio is None:
        raise RuntimeError("Seed-VC convert_voice returned no audio")

    # unwrap (sr, array) tuple if present
    if isinstance(audio, (tuple, list)) and len(audio) == 2:
        _, audio = audio

    if audio.ndim == 1:
        audio = audio[:, None]
    elif audio.ndim > 2:
        audio = audio.squeeze()
        if audio.ndim == 1:
            audio = audio[:, None]
    _sf.write(str(out), audio, 22050)


def _dispatch_generate(backend: str, text: str, ref_path: Path | None, out: Path, p1: float, p2: float, ref_text: str) -> None:
    if backend == "xtts_v2":
        _xtts_generate(text, ref_path, out, temperature=p1, speed=p2)
    elif backend == "f5_tts":
        _f5_generate(text, ref_path, out, steps=p1, speed=p2, ref_text=ref_text)
    elif backend == "chatterbox":
        _chatterbox_generate(text, ref_path, out, exaggeration=p1, cfg_weight=p2)
    else:
        raise ValueError(f"Unknown backend: {backend}")


def _ideal_max_chars(backend: str, ref_path: Path | None) -> int:
    """Per-backend character budget used to decide when a line needs auto-splitting.

    F5-TTS predicts output duration from the ref:gen length ratio (see its text_hint),
    so its budget scales with the actual reference clip instead of a flat number —
    ~15 chars/sec of speech, allowed to run up to ~2.5x the reference duration.
    """
    if backend == "f5_tts" and ref_path is not None:
        try:
            dur = sf.info(str(ref_path)).duration
            return max(80, int(dur * 15 * 2.5))
        except Exception:
            pass
    return BACKENDS[backend].get("split_chars", 250)


def _split_text_for_generation(text: str, max_chars: int) -> list[str]:
    """Greedily pack sentences into chunks no longer than max_chars.

    A sentence longer than max_chars on its own is further broken on word
    boundaries so no chunk is ever left unbounded.
    """
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return [text]

    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks: list[str] = []
    current = ""
    for sent in sentences:
        if len(sent) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            piece = ""
            for word in sent.split(" "):
                candidate = f"{piece} {word}".strip()
                if len(candidate) > max_chars and piece:
                    chunks.append(piece)
                    piece = word
                else:
                    piece = candidate
            current = piece
            continue
        candidate = f"{current} {sent}".strip()
        if len(candidate) > max_chars and current:
            chunks.append(current)
            current = sent
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


_CROSSFADE_SECONDS = 0.04
_TRIM_KEEP_LEAD = 0.03     # seconds of lead-in kept before the first speech when trimming a chunk
_TRIM_KEEP_TAIL = 0.08     # ...and after the last speech (so word endings aren't clipped)
# Pause put between two glued chunks, by how the earlier chunk's text ends. The model's own
# silence at a chunk's edges is trimmed away, so this is the only gap — flowing, but still
# respecting the punctuation.
_PAUSE_AFTER = {".": 0.35, "!": 0.35, "?": 0.35, "…": 0.4, ":": 0.25, ";": 0.25, ",": 0.18, "—": 0.18}


def _trim_edges(data: "np.ndarray", sr: int) -> "np.ndarray":
    """Cut silence off the start and end of a generated chunk (shape [frames, channels]).
    'Silence' = frames more than 40 dB under the chunk's loudest, so quiet word endings and
    breaths inside the speech stay."""
    import numpy as np
    if len(data) == 0:
        return data
    mono = np.abs(data).max(axis=1) if data.ndim > 1 else np.abs(data)
    win = max(int(0.01 * sr), 1)
    n = len(mono) // win
    if n == 0:
        return data
    env = mono[: n * win].reshape(n, win).max(axis=1)
    peak = env.max()
    if peak <= 1e-4:
        return data
    live = np.nonzero(env > peak * 0.01)[0]          # -40 dB
    if len(live) == 0:
        return data
    start = max(int(live[0] * win - _TRIM_KEEP_LEAD * sr), 0)
    end = min(int((live[-1] + 1) * win + _TRIM_KEEP_TAIL * sr), len(data))
    return data[start:end]


def _pause_after(text: str) -> float:
    t = text.rstrip().rstrip("\"'”’)]»")
    return _PAUSE_AFTER.get(t[-1:], 0.0)
_CHARS_PER_SECOND = 15.0  # rough speaking-rate estimate, used only to size the preview bar


def _generate_chunked(
    chunks: list[str], backend: str, ref_path: Path | None, out: Path,
    p1: float, p2: float, ref_text: str,
    progress_ref: list | None, preview_ref: dict | None = None,
) -> None:
    """Generate each chunk to a scratch file and blend it into `out` on disk with
    a short crossfade at each seam. Only the current and previous chunk's audio is
    ever held in memory — bounded, not growing with total length.

    After every chunk, `out` is closed and reopened rather than kept open for the
    whole batch, so its WAV header is finalized each time: `out` is a valid, fully
    playable clip of everything generated so far at any point during the batch,
    not just once the whole thing is done. preview_ref (if given) is updated in
    place with {"path", "ready_seconds", "estimated_total_seconds", "done"} so a
    UI can poll it to drive a buffering-style preview.
    """
    import numpy as np

    if preview_ref is not None:
        preview_ref.clear()
        preview_ref.update(
            path=str(out), done=False, ready_seconds=0.0,
            estimated_total_seconds=sum(len(c) for c in chunks) / _CHARS_PER_SECOND,
        )

    scratch_dir = Path(tempfile.mkdtemp(prefix="faily_batch_"))
    pending: "np.ndarray | None" = None
    prev_text = ""
    sr: int | None = None
    channels: int | None = None
    frames_written = 0
    wrote_any = False

    def _flush(segment) -> None:
        nonlocal frames_written, wrote_any
        if wrote_any:
            f = sf.SoundFile(str(out), mode="r+")
            f.seek(0, sf.SEEK_END)
        else:
            f = sf.SoundFile(str(out), mode="w", samplerate=sr, channels=channels)
        try:
            f.write(segment)
        finally:
            f.close()
        wrote_any = True
        frames_written += len(segment)
        if preview_ref is not None:
            preview_ref["ready_seconds"] = frames_written / sr

    try:
        for i, chunk in enumerate(chunks):
            if progress_ref is not None:
                progress_ref[0] = i / len(chunks)
            chunk_path = scratch_dir / f"chunk_{i:03d}.wav"
            _dispatch_generate(backend, chunk, ref_path, chunk_path, p1, p2, ref_text)
            data, chunk_sr = sf.read(str(chunk_path), dtype="float32", always_2d=True)
            if sr is None:
                sr, channels = chunk_sr, data.shape[1]
            data = _trim_edges(data, chunk_sr)

            if pending is None:
                pending = data
                prev_text = chunk
                continue

            gap = int(_pause_after(prev_text) * sr)
            prev_text = chunk
            if gap > 0:      # a punctuation pause: deliberate silence instead of whatever the model left
                _flush(np.concatenate([pending, np.zeros((gap, pending.shape[1]), dtype="float32")]))
                pending = data
                continue

            fade = min(int(_CROSSFADE_SECONDS * sr), len(pending), len(data))
            if fade > 0:
                fade_out = np.linspace(1.0, 0.0, fade, dtype="float32")[:, None]
                fade_in = np.linspace(0.0, 1.0, fade, dtype="float32")[:, None]
                blended = pending[-fade:] * fade_out + data[:fade] * fade_in
                _flush(np.concatenate([pending[:-fade], blended]))
                pending = data[fade:]
            else:
                _flush(pending)
                pending = data
        if pending is not None:
            _flush(pending)
    finally:
        shutil.rmtree(scratch_dir, ignore_errors=True)
        if preview_ref is not None:
            preview_ref["done"] = True
            if sr:
                preview_ref["estimated_total_seconds"] = frames_written / sr


def generate(
    text: str,
    ref_path: Path | None,
    progress_ref: list | None = None,
    output_dir: Path | None = None,
    backend: str = "xtts_v2",
    param1: float | None = None,
    param2: float | None = None,
    ref_text: str = "",
    char_name: str | None = None,
    preview_ref: dict | None = None,
) -> Path:
    if output_dir is None:
        output_dir = history_dir(char_name, ref_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = BACKENDS[backend]
    p1 = param1 if param1 is not None else cfg["param1"]["default"]
    p2 = param2 if param2 is not None else cfg["param2"]["default"]

    if progress_ref is not None:
        progress_ref[0] = 0.2

    out = output_dir / _make_clip_name(char_name, backend, text, output_dir)
    chunks = _split_text_for_generation(text, _ideal_max_chars(backend, ref_path))

    if len(chunks) == 1:
        if progress_ref is not None:
            progress_ref[0] = 0.4
        _dispatch_generate(backend, text, ref_path, out, p1, p2, ref_text)
    else:
        _generate_chunked(chunks, backend, ref_path, out, p1, p2, ref_text, progress_ref, preview_ref)

    if progress_ref is not None:
        progress_ref[0] = 1.0

    ensure_stereo(out)
    if preview_ref is not None:
        preview_ref["done"] = True
    return out


