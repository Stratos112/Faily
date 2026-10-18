"""
Patches piper_train/__main__.py to honor FAILY_PIPER_FREEZE: a comma-separated
list of generator submodule names to freeze during fine-tuning
(e.g. "enc_p,dp").

Why: fine-tuning a pretrained Piper voice on a small, different-speaker
dataset drags the text encoder (enc_p: text -> pronunciation) and duration
predictor (dp: pacing) along with the voice. Within ~10 epochs word error
rate on unseen text goes from ~0 to 0.2-0.3 and speech rushes. Freezing them
keeps the base model's pronunciation (and pacing, if dp is frozen) while the
decoder / flow / posterior encoder still learn the new voice.

Frozen parameters get no gradient, so Adam skips them; their weights stay
bit-identical to the checkpoint's. No effect unless FAILY_PIPER_FREEZE is set.

Idempotent — safe to run every setup. Needs patch_piper_train_stop_request.py
applied first (the Callback import):
    piper_venv/bin/python scripts/patch_piper_train_freeze.py
"""
import importlib.util
import pathlib

_spec = importlib.util.find_spec("piper_train")
if _spec is None:
    raise RuntimeError("piper_train is not installed in the current Python environment")
PIPER_TRAIN_DIR = pathlib.Path(_spec.submodule_search_locations[0])

main_py = PIPER_TRAIN_DIR / "__main__.py"
text = main_py.read_text(encoding="utf-8")

MARKER = "FAILY_PIPER_FREEZE"

CALLBACK = '''
class _FailyFreeze(Callback):  # FAILY_PIPER_FREEZE
    """Freeze generator submodules named in $FAILY_PIPER_FREEZE (e.g. "enc_p,dp")."""

    def __init__(self, prefixes):
        self.prefixes = tuple(p.strip() + "." for p in prefixes.split(",") if p.strip())

    def on_train_start(self, trainer, pl_module):
        frozen = 0
        for name, param in pl_module.model_g.named_parameters():
            if name.startswith(self.prefixes):
                param.requires_grad = False
                frozen += param.numel()
        _LOGGER.info("Frozen %s: %.1fM parameters", ", ".join(self.prefixes), frozen / 1e6)
        if not frozen:
            raise RuntimeError(f"FAILY_PIPER_FREEZE matched no parameters: {self.prefixes}")

'''

DEF_OLD = "\ndef main():\n"
USE_OLD = "    dict_args = vars(args)\n"
USE_NEW = (
    "    if os.environ.get(\"FAILY_PIPER_FREEZE\"):\n"
    "        trainer.callbacks.append(_FailyFreeze(os.environ[\"FAILY_PIPER_FREEZE\"]))\n\n"
    + USE_OLD
)

if MARKER in text:
    print("freeze patch: already applied")
else:
    for needle in (DEF_OLD, USE_OLD):
        if text.count(needle) != 1:
            raise RuntimeError(
                f"Could not find expected {needle!r} in {main_py} — "
                "piper-train's __main__.py may have changed shape."
            )
    if "Callback" not in text.split("def main")[0] or "\nimport os\n" not in text:
        raise RuntimeError(
            "Callback/os not imported in piper_train/__main__.py — run "
            "patch_piper_train_stop_request.py and patch_piper_train_learning_rate.py first."
        )
    text = text.replace(DEF_OLD, CALLBACK + DEF_OLD, 1)
    text = text.replace(USE_OLD, USE_NEW, 1)
    main_py.write_text(text, encoding="utf-8")
    print(f"freeze patch: applied to {main_py}")

cache = PIPER_TRAIN_DIR / "__pycache__"
if cache.exists():
    for pyc in cache.glob("__main__.cpython-*.pyc"):
        pyc.unlink()
