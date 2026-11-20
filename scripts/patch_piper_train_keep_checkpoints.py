"""
Patches piper_train/__main__.py to keep EVERY periodic checkpoint.

piper_train builds its saver as ModelCheckpoint(every_n_epochs=N) with Lightning's
default save_top_k=1. There is no monitored metric, and in that case Lightning 1.7
deletes the previous checkpoint each time it saves a new one
(ModelCheckpoint._save_none_monitor_checkpoint), so a run only ever left its last
checkpoint behind (plus the "stop early" one) and the audition list had one entry.
save_top_k=-1 keeps them all. Faily's train() wipes piper_train/ at the start of each
run and spaces the saves so a run keeps at most ~20 (piper.checkpoint_interval).

Idempotent — safe to run every setup.

Run after installing piper-train:
    piper_venv/bin/python scripts/patch_piper_train_keep_checkpoints.py
"""
import importlib.util
import pathlib

_spec = importlib.util.find_spec("piper_train")
if _spec is None:
    raise RuntimeError("piper_train is not installed in the current Python environment")
PIPER_TRAIN_DIR = pathlib.Path(_spec.submodule_search_locations[0])

main_py = PIPER_TRAIN_DIR / "__main__.py"
text = main_py.read_text(encoding="utf-8")

MARKER = "FAILY_KEEP_CHECKPOINTS"
OLD = "ModelCheckpoint(every_n_epochs=args.checkpoint_epochs)"
NEW = f"ModelCheckpoint(every_n_epochs=args.checkpoint_epochs, save_top_k=-1)]  # {MARKER}\n"

if MARKER in text:
    print("keep-checkpoints patch: already applied")
else:
    line_old = OLD + "]\n"
    if text.count(line_old) != 1:
        raise RuntimeError(
            f"Could not find expected {line_old!r} in {main_py} — "
            "piper-train's __main__.py may have changed shape."
        )
    main_py.write_text(text.replace(line_old, NEW), encoding="utf-8")
    print(f"keep-checkpoints patch: applied to {main_py}")
