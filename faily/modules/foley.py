import numpy as np
import soundfile as sf
from datetime import datetime
from pathlib import Path
from faily.core.model_manager import manager, SFX_MODELS_DIR


def _patch_model_output():
    """
    diffusers AudioLDM2 slices text-encoder ModelOutputs with [:, None, :].
    transformers 5.x ModelOutput.__getitem__ doesn't support tuple indices.
    Patch it once — cache-proof, called before every inference.
    Prefers text_embeds → pooler_output (CLAP projected 512-dim in transformers 5.x) → last_hidden_state.
    """
    try:
        import transformers.utils.generic as _g
        import torch as _t
    except ImportError:
        return
    if getattr(_g.ModelOutput, "_faily_patched", False):
        return
    _orig = _g.ModelOutput.__getitem__
    def _getitem(self, k):
        if isinstance(k, tuple):
            for _attr in ("text_embeds", "pooler_output", "last_hidden_state"):
                v = getattr(self, _attr, None)
                if isinstance(v, _t.Tensor):
                    return v[k]
            for v in self.to_tuple():
                if isinstance(v, _t.Tensor):
                    return v[k]
        return _orig(self, k)
    _g.ModelOutput.__getitem__ = _getitem
    _g.ModelOutput._faily_patched = True


_BUNDLED: dict[str, str] = {
    "AudioLDM2":            "cvssp/audioldm2",
    "AudioLDM2 Large":      "cvssp/audioldm2-large",
    "Tango 2":              "declare-lab/tango2",
    "Stable Audio Open":    "stabilityai/stable-audio-open-1.0",
    "AudioGen Medium":      "facebook/audiogen-medium",
    "AudioGen Large":       "facebook/audiogen-large",
}

_STABLE_AUDIO = {"stabilityai/stable-audio-open-1.0"}
_AUDIOGEN = {"facebook/audiogen-medium", "facebook/audiogen-large"}

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


# ── AudioLDM2 / Tango 2 ───────────────────────────────────────────────────────

def _patch_generate_lm(pipe):
    """Windows diffusers uses output.last_hidden_state which doesn't exist on
    CausalLMOutputWithCrossAttentions; patch to use output.hidden_states[-1].
    Also seeds cache_position for transformers 5.x KV-cache handling.
    Called after every manager.load() — guarded by flag on the pipe object."""
    if getattr(pipe, "_faily_lm_patched", False):
        return
    import types
    import torch as _t

    def _prep(inputs_embeds, attention_mask=None, past_key_values=None, cache_position=None, **kw):
        if past_key_values is not None:
            inputs_embeds = inputs_embeds[:, -1:]
        result = {"inputs_embeds": inputs_embeds, "attention_mask": attention_mask,
                  "past_key_values": past_key_values, "use_cache": kw.get("use_cache")}
        if cache_position is not None:
            result["cache_position"] = cache_position
        return result

    def generate_language_model(self, inputs_embeds=None, max_new_tokens=8, **model_kwargs):
        max_new_tokens = max_new_tokens if max_new_tokens is not None else self.language_model.config.max_new_tokens
        if "cache_position" not in model_kwargs:
            model_kwargs["cache_position"] = _t.arange(
                inputs_embeds.shape[1], device=inputs_embeds.device
            )
        for _ in range(max_new_tokens):
            model_inputs = _prep(inputs_embeds, **model_kwargs)
            output = self.language_model(**model_inputs, output_hidden_states=True, return_dict=True)
            next_hidden_states = output.hidden_states[-1]
            inputs_embeds = _t.cat([inputs_embeds, next_hidden_states[:, -1:, :]], dim=1)
            model_kwargs = self.language_model._update_model_kwargs_for_generation(output, model_kwargs)
        return inputs_embeds[:, -max_new_tokens:, :]

    pipe.generate_language_model = types.MethodType(generate_language_model, pipe)
    pipe._faily_lm_patched = True


def _loader_audioldm2(model_id: str):
    # diffusers < 0.33 imports FLAX_WEIGHTS_NAME from transformers.utils which was removed in transformers 5.x
    import transformers.utils as _tu
    if not hasattr(_tu, "FLAX_WEIGHTS_NAME"):
        _tu.FLAX_WEIGHTS_NAME = "flax_model.msgpack"

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
    _patch_generate_lm(pipe)
    return pipe.to(manager.device)


def _generate_audioldm2(pipe, prompt, duration, steps, guidance, candidates, progress_ref):
    _patch_generate_lm(pipe)

    def _cb(step, _ts, _lat):
        if progress_ref is not None:
            progress_ref[0] = (step + 1) / steps

    result = pipe(
        prompt,
        num_inference_steps=steps,
        audio_length_in_s=duration,
        guidance_scale=guidance,
        num_waveforms_per_prompt=candidates,
        callback=_cb,
        callback_steps=1,
    )
    return result.audios[0], 16000


# ── Stable Audio Open ─────────────────────────────────────────────────────────

def _loader_stable_audio(model_id: str):
    from diffusers import StableAudioPipeline
    import torch
    dtype = torch.float16 if manager.device == "cuda" else torch.float32
    pipe = StableAudioPipeline.from_pretrained(
        model_id, torch_dtype=dtype, cache_dir=str(SFX_MODELS_DIR)
    )
    return pipe.to(manager.device)


def _generate_stable_audio(pipe, prompt, duration, steps, guidance, candidates, progress_ref):
    def _cb(step, _ts, _lat):
        if progress_ref is not None:
            progress_ref[0] = (step + 1) / steps

    result = pipe(
        prompt,
        audio_end_in_s=duration,
        num_inference_steps=steps,
        guidance_scale=guidance,
        num_waveforms_per_prompt=candidates,
        callback=_cb,
        callback_steps=1,
    )
    sr = pipe.vae.sampling_rate
    # audios: (batch, channels, samples) — mix to mono
    audio = result.audios[0].mean(axis=0) if result.audios[0].ndim > 1 else result.audios[0]
    return audio, sr


# ── AudioGen ──────────────────────────────────────────────────────────────────

def _loader_audiogen(model_id: str):
    from transformers import AutoProcessor, MusicgenForConditionalGeneration
    processor = AutoProcessor.from_pretrained(model_id, cache_dir=str(SFX_MODELS_DIR))
    model = MusicgenForConditionalGeneration.from_pretrained(
        model_id, cache_dir=str(SFX_MODELS_DIR)
    )
    return (processor, model.to(manager.device))


def _generate_audiogen(pipe_tuple, prompt, duration, candidates, progress_ref):
    import torch
    processor, model = pipe_tuple
    sr = model.config.audio_encoder.sampling_rate
    max_new_tokens = int(duration * 50)  # AudioGen uses ~50 tokens/sec

    if progress_ref is not None:
        progress_ref[0] = 0.1
    inputs = processor(text=[prompt], padding=True, return_tensors="pt").to(manager.device)
    with torch.no_grad():
        audio = model.generate(**inputs, max_new_tokens=max_new_tokens)
    if progress_ref is not None:
        progress_ref[0] = 1.0

    # audio: (batch, channels, samples)
    wav = audio[0, 0].cpu().numpy().astype(np.float32)
    return wav, sr


# ── dispatch ──────────────────────────────────────────────────────────────────

def generate(
    prompt: str,
    model_id: str,
    duration: float = 5.0,
    steps: int = 50,
    guidance: float = 3.5,
    progress_ref: list | None = None,
    output_dir: Path | None = None,
    candidates: int = 1,
    normalize: bool = False,
    fade: float = 0.0,
) -> Path:
    _patch_model_output()
    if output_dir is None:
        output_dir = SFX_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    if model_id in _STABLE_AUDIO:
        pipe = manager.load(model_id, lambda: _loader_stable_audio(model_id))
        audio, sr = _generate_stable_audio(pipe, prompt, duration, steps, guidance, candidates, progress_ref)
    elif model_id in _AUDIOGEN:
        pipe_tuple = manager.load(model_id, lambda: _loader_audiogen(model_id))
        audio, sr = _generate_audiogen(pipe_tuple, prompt, duration, candidates, progress_ref)
    else:
        pipe = manager.load(model_id, lambda: _loader_audioldm2(model_id))
        audio, sr = _generate_audioldm2(pipe, prompt, duration, steps, guidance, candidates, progress_ref)

    if normalize:
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio / peak

    if fade > 0.0:
        fade_samples = min(int(fade * sr), len(audio) // 2)
        if fade_samples > 0:
            ramp = np.linspace(0.0, 1.0, fade_samples)
            audio[:fade_samples] *= ramp
            audio[-fade_samples:] *= ramp[::-1]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = output_dir / f"sfx_{ts}.wav"
    sf.write(str(out), audio, sr)
    return out
