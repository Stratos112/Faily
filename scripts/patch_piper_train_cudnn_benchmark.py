"""
Patches piper_train/__main__.py to set torch.backends.cudnn.benchmark = False.

piper_train hard-codes benchmark=True. VITS batches have a different
spectrogram length every step, so cuDNN re-benchmarks every convolution for
every new shape. On Blackwell (sm_120, cu128) that planning is CPU-bound and
costs ~4 s per generator step while the GPU sits ~idle (60 ms of real work) —
an epoch on a few hundred clips takes ~10 minutes instead of ~15 seconds, with
no progress output to explain why.

Idempotent — safe to run every setup.

Run after installing piper-train:
    piper_venv/bin/python scripts/patch_piper_train_cudnn_benchmark.py
"""
import importlib.util
import pathlib

_spec = importlib.util.find_spec("piper_train")
if _spec is None:
    raise RuntimeError("piper_train is not installed in the current Python environment")
PIPER_TRAIN_DIR = pathlib.Path(_spec.submodule_search_locations[0])

main_py = PIPER_TRAIN_DIR / "__main__.py"
text = main_py.read_text(encoding="utf-8")

OLD = "torch.backends.cudnn.benchmark = True"
NEW = "torch.backends.cudnn.benchmark = False"

if NEW in text:
    print("cudnn.benchmark=False patch: already applied")
elif OLD not in text:
    raise RuntimeError(
        f"Could not find expected '{OLD}' in {main_py} — "
        "piper-train's __main__.py may have changed shape."
    )
else:
    main_py.write_text(text.replace(OLD, NEW, 1), encoding="utf-8")
    print(f"cudnn.benchmark=False patch: applied to {main_py}")

cache = PIPER_TRAIN_DIR / "__pycache__"
if cache.exists():
    for pyc in cache.glob("__main__.cpython-*.pyc"):
        pyc.unlink()
