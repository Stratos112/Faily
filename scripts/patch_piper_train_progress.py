"""
Patches piper_train/__main__.py to write <default_root_dir>/progress.json.

When piper_train's stdout is a pipe (as it is under Faily), Lightning prints no
progress bar at all, so there is nothing for the app to parse. This adds a
callback that records the current epoch, the batch within it, and the latest
generator/discriminator/validation losses, rewritten atomically at most every
couple of seconds and at every epoch end. Faily polls the file for its
progress display.

Idempotent — safe to run every setup. Run after patch_piper_train_stop_request.py
(it needs the `Callback` import that patch adds):
    piper_venv/bin/python scripts/patch_piper_train_progress.py
"""
import importlib.util
import pathlib

_spec = importlib.util.find_spec("piper_train")
if _spec is None:
    raise RuntimeError("piper_train is not installed in the current Python environment")
PIPER_TRAIN_DIR = pathlib.Path(_spec.submodule_search_locations[0])

main_py = PIPER_TRAIN_DIR / "__main__.py"
text = main_py.read_text(encoding="utf-8")

MARKER = "FAILY_PROGRESS"

CALLBACK = '''
class _FailyProgress(Callback):  # FAILY_PROGRESS
    """Write training progress to <default_root_dir>/progress.json."""

    def __init__(self):
        self._last = 0.0

    def _write(self, trainer, batch):
        import time

        metrics = trainer.callback_metrics

        def val(key):
            v = metrics.get(key)
            return float(v) if v is not None else None

        path = Path(trainer.default_root_dir) / "progress.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({
            "epoch": trainer.current_epoch,
            "batch": batch,
            "batches": trainer.num_training_batches,
            "gen": val("loss_gen_all"),
            "disc": val("loss_disc_all"),
            "val": val("val_loss"),
            "time": time.time(),
        }))
        tmp.replace(path)

    def on_train_start(self, trainer, pl_module):
        self._write(trainer, 0)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, *args):
        import time

        if time.time() - self._last >= 2.0:
            self._last = time.time()
            self._write(trainer, batch_idx + 1)

    def on_train_epoch_end(self, trainer, pl_module):
        self._write(trainer, trainer.num_training_batches)

'''

DEF_OLD = "\ndef main():\n"
USE_OLD = "    dict_args = vars(args)\n"
USE_NEW = "    trainer.callbacks.append(_FailyProgress())\n\n" + USE_OLD

if MARKER in text:
    print("progress patch: already applied")
else:
    for needle in (DEF_OLD, USE_OLD):
        if text.count(needle) != 1:
            raise RuntimeError(
                f"Could not find expected {needle!r} in {main_py} — "
                "piper-train's __main__.py may have changed shape."
            )
    if "Callback" not in text.split("def main")[0]:
        raise RuntimeError(
            "Callback is not imported in piper_train/__main__.py — run "
            "patch_piper_train_stop_request.py first."
        )
    text = text.replace(DEF_OLD, CALLBACK + DEF_OLD, 1)
    text = text.replace(USE_OLD, USE_NEW, 1)
    main_py.write_text(text, encoding="utf-8")
    print(f"progress patch: applied to {main_py}")

cache = PIPER_TRAIN_DIR / "__pycache__"
if cache.exists():
    for pyc in cache.glob("__main__.cpython-*.pyc"):
        pyc.unlink()
