"""
Patches piper_train/vits/lightning.py's VitsModel to default num_workers=0
instead of 1.

piper-train never exposes --num-workers on the CLI (add_model_specific_args
in lightning.py doesn't declare it, and __main__.py's dict_args is just
vars(args)), so the VitsModel.__init__ default of num_workers=1 is the value
actually used for every train/val/test DataLoader, unconditionally, with no
way to override it from outside the package.

On Windows, num_workers=1 means PyTorch's DataLoader spawns one real worker
subprocess (Windows has no fork, only spawn) the first time it's iterated —
i.e. right when trainer.fit() starts pulling the first training batch, which
is exactly where training has been crashing with a native access violation
(exit 3221225477 / 0xC0000005) and zero Python traceback, even with
CUDA_LAUNCH_BLOCKING=1 producing no CUDA-side error first. That combination
(silent native crash, no CUDA diagnostic output, right at first-batch spawn)
points at the DataLoader worker subprocess itself rather than a CUDA kernel.
num_workers=0 runs data loading in-process — slower, but it removes the
spawn entirely, which is the only lever available since there's no CLI flag.

Idempotent — safe to run every setup.

Run after installing piper-train:
    <piper_venv>\\Scripts\\python scripts\\patch_piper_train_num_workers.py
"""
import importlib.util
import pathlib

_spec = importlib.util.find_spec("piper_train")
if _spec is None:
    raise RuntimeError("piper_train is not installed in the current Python environment")
PIPER_TRAIN_DIR = pathlib.Path(_spec.submodule_search_locations[0])

lightning_py = PIPER_TRAIN_DIR / "vits" / "lightning.py"
text = lightning_py.read_text(encoding="utf-8")

OLD = "        num_workers: int = 1,\n"
NEW = "        num_workers: int = 0,\n"

if "num_workers: int = 0," in text:
    print("num_workers=0 (no DataLoader worker subprocess) patch: already applied")
elif OLD not in text:
    raise RuntimeError(
        f"Could not find expected 'num_workers: int = 1,' in {lightning_py} — "
        "piper-train's lightning.py may have changed shape."
    )
else:
    text = text.replace(OLD, NEW, 1)
    lightning_py.write_text(text, encoding="utf-8")
    print(f"num_workers=0 (no DataLoader worker subprocess) patch: applied to {lightning_py}")

cache = PIPER_TRAIN_DIR / "vits" / "__pycache__"
if cache.exists():
    for pyc in cache.glob("lightning.cpython-*.pyc"):
        pyc.unlink()
