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


def _patch_generate_lm(pipe):
    """Windows diffusers uses output.last_hidden_state which doesn't exist on
    CausalLMOutputWithCrossAttentions; patch to use output.hidden_states[-1].
    Called after every manager.load() — guarded by flag on the pipe object."""
    if getattr(pipe, "_faily_lm_patched", False):
        return
    import types
    import torch as _t

    def _prep(inputs_embeds, attention_mask=None, past_key_values=None, **kw):
        if past_key_values is not None:
            inputs_embeds = inputs_embeds[:, -1:]
        return {"inputs_embeds": inputs_embeds, "attention_mask": attention_mask,
                "past_key_values": past_key_values, "use_cache": kw.get("use_cache")}

    def generate_language_model(self, inputs_embeds=None, max_new_tokens=8, **model_kwargs):
        max_new_tokens = max_new_tokens if max_new_tokens is not None else self.language_model.config.max_new_tokens
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

    pipe = manager.load(model_id, lambda: _loader_audioldm2(model_id))
    _patch_generate_lm(pipe)

    def _cb(step: int, timestep: int, latents):
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

    audio: np.ndarray = result.audios[0]

    if normalize:
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio / peak

    if fade > 0.0:
        sr = 16000
        fade_samples = min(int(fade * sr), len(audio) // 2)
        if fade_samples > 0:
            ramp = np.linspace(0.0, 1.0, fade_samples)
            audio[:fade_samples] *= ramp
            audio[-fade_samples:] *= ramp[::-1]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = output_dir / f"sfx_{ts}.wav"
    sf.write(str(out), audio, 16000)
    return out
