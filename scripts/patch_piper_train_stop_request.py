"""
Patches piper_train/__main__.py so a running fine-tune can be stopped cleanly.

Faily's "stop early" drops a STOP_REQUEST file into the trainer's
--default_root_dir. This adds a Lightning callback that, at the end of the
current epoch, sees the file, saves a checkpoint of that exact epoch next to
the regular ones, and stops training. Without it the only way to stop is to
kill the process, which loses everything since the last periodic checkpoint
(--checkpoint-epochs, typically up to 100 epochs).

Idempotent — safe to run every setup.

Run after installing piper-train:
    piper_venv/bin/python scripts/patch_piper_train_stop_request.py
"""
import importlib.util
import pathlib

_spec = importlib.util.find_spec("piper_train")
if _spec is None:
    raise RuntimeError("piper_train is not installed in the current Python environment")
PIPER_TRAIN_DIR = pathlib.Path(_spec.submodule_search_locations[0])

main_py = PIPER_TRAIN_DIR / "__main__.py"
text = main_py.read_text(encoding="utf-8")

MARKER = "FAILY_STOP_REQUEST"

CALLBACK = '''
class _FailyStopRequest(Callback):  # FAILY_STOP_REQUEST
    """Stop cleanly, with a checkpoint, once <default_root_dir>/STOP_REQUEST exists."""

    def on_train_epoch_end(self, trainer, pl_module):
        flag = Path(trainer.default_root_dir) / "STOP_REQUEST"
        if not flag.exists():
            return
        ckpt_dir = Path(trainer.checkpoint_callback.dirpath)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        path = ckpt_dir / f"epoch={trainer.current_epoch}-step={trainer.global_step}.ckpt"
        trainer.save_checkpoint(str(path))
        _LOGGER.info("Stop requested: saved %s", path)
        flag.unlink(missing_ok=True)
        trainer.should_stop = True

'''

IMPORT_OLD = "from pytorch_lightning.callbacks import ModelCheckpoint\n"
IMPORT_NEW = "from pytorch_lightning.callbacks import Callback, ModelCheckpoint\n"
DEF_OLD = "\ndef main():\n"
USE_OLD = "    dict_args = vars(args)\n"
USE_NEW = "    trainer.callbacks.append(_FailyStopRequest())\n\n" + USE_OLD

if MARKER in text:
    print("stop-request patch: already applied")
else:
    for needle in (IMPORT_OLD, DEF_OLD, USE_OLD):
        if text.count(needle) != 1:
            raise RuntimeError(
                f"Could not find expected {needle!r} in {main_py} — "
                "piper-train's __main__.py may have changed shape."
            )
    text = text.replace(IMPORT_OLD, IMPORT_NEW, 1)
    text = text.replace(DEF_OLD, CALLBACK + DEF_OLD, 1)
    text = text.replace(USE_OLD, USE_NEW, 1)
    main_py.write_text(text, encoding="utf-8")
    print(f"stop-request patch: applied to {main_py}")

cache = PIPER_TRAIN_DIR / "__pycache__"
if cache.exists():
    for pyc in cache.glob("__main__.cpython-*.pyc"):
        pyc.unlink()
