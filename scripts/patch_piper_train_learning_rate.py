"""
Patches piper_train/__main__.py to honor FAILY_PIPER_LR (a learning rate).

piper_train has no learning-rate flag, and on --resume_from_checkpoint
Lightning restores the optimizer state from the checkpoint, so even a changed
default would be overwritten by the checkpoint's saved rate (the lessac base
checkpoint is already decayed from 2e-4 to ~1.5e-4, and decays further every
epoch, gamma=0.999875). This adds a callback that, once the checkpoint has
been restored and training is about to start, sets that rate on both
optimizers and their schedulers. The ExponentialLR decay then continues from
the new value. No effect unless FAILY_PIPER_LR is set.

Idempotent — safe to run every setup.

Run after installing piper-train:
    piper_venv/bin/python scripts/patch_piper_train_learning_rate.py
"""
import importlib.util
import pathlib

_spec = importlib.util.find_spec("piper_train")
if _spec is None:
    raise RuntimeError("piper_train is not installed in the current Python environment")
PIPER_TRAIN_DIR = pathlib.Path(_spec.submodule_search_locations[0])

main_py = PIPER_TRAIN_DIR / "__main__.py"
text = main_py.read_text(encoding="utf-8")

MARKER = "FAILY_PIPER_LR"

CALLBACK = '''
class _FailyLearningRate(Callback):  # FAILY_PIPER_LR
    """Force the learning rate to $FAILY_PIPER_LR after the checkpoint restore."""

    def __init__(self, lr):
        self.lr = lr

    def on_train_start(self, trainer, pl_module):
        for opt in trainer.optimizers:
            for group in opt.param_groups:
                _LOGGER.info("Learning rate %.3g -> %.3g", group["lr"], self.lr)
                group["lr"] = self.lr
        for cfg in trainer.lr_scheduler_configs:
            sched = cfg.scheduler
            sched.base_lrs = [self.lr] * len(sched.base_lrs)
            sched._last_lr = [self.lr] * len(sched.base_lrs)

'''

IMPORT_OLD = "from pytorch_lightning.callbacks import"
DEF_OLD = "\ndef main():\n"
USE_OLD = "    dict_args = vars(args)\n"
USE_NEW = (
    "    if os.environ.get(\"FAILY_PIPER_LR\"):\n"
    "        trainer.callbacks.append(_FailyLearningRate(float(os.environ[\"FAILY_PIPER_LR\"])))\n\n"
    + USE_OLD
)

if MARKER in text:
    print("learning-rate patch: already applied")
else:
    for needle in (DEF_OLD, USE_OLD):
        if text.count(needle) != 1:
            raise RuntimeError(
                f"Could not find expected {needle!r} in {main_py} — "
                "piper-train's __main__.py may have changed shape."
            )
    if IMPORT_OLD not in text or "Callback" not in text.split("def main")[0]:
        raise RuntimeError(
            "Callback is not imported in piper_train/__main__.py — run "
            "patch_piper_train_stop_request.py first."
        )
    text = text.replace(DEF_OLD, CALLBACK + DEF_OLD, 1)
    text = text.replace(USE_OLD, USE_NEW, 1)
    if "\nimport os\n" not in text:
        text = text.replace("import logging\n", "import logging\nimport os\n", 1)
    main_py.write_text(text, encoding="utf-8")
    print(f"learning-rate patch: applied to {main_py}")

cache = PIPER_TRAIN_DIR / "__pycache__"
if cache.exists():
    for pyc in cache.glob("__main__.cpython-*.pyc"):
        pyc.unlink()
