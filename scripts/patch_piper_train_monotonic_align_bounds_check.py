"""
Adds a bounds assertion to piper_train/vits/monotonic_align/__init__.py's
maximum_path(), right before it calls into the native maximum_path_c extension.

Training crashes on the very first batch with a silent native access
violation (exit 0xC0000005, zero Python traceback) inside maximum_path_c,
called from maximum_path() (see faulthandler output: models.py:647 ->
monotonic_align/__init__.py:19). Both maximum_path_c and maximum_path_each in
core.pyx are compiled with @cython.boundscheck(False) and run nogil, so if
t_t_max/t_s_max (computed from this batch's padding mask) ever come out
larger than the actual path/neg_cent array dimensions for that batch, the
native indexing silently corrupts memory instead of raising — exactly the
"crash with zero Python traceback" shape we've observed. Re-enabling bounds
checking inside the compiled extension itself runs into Cython's nogil/void
exception-propagation rules (maximum_path_each returns void with no `except *`
clause, so an exception raised during a nogil bounds check isn't guaranteed to
propagate cleanly). Asserting the same relationship in the pure-Python
wrapper, before the native call, sidesteps all of that and turns a theorized
out-of-bounds native crash into a normal, catchable AssertionError with the
exact offending shapes — diagnostic only; remove once the real bug is found.

Idempotent — safe to run every setup.

Run after installing piper-train:
    <piper_venv>\\Scripts\\python scripts\\patch_piper_train_monotonic_align_bounds_check.py
"""
import importlib.util
import pathlib

_spec = importlib.util.find_spec("piper_train")
if _spec is None:
    raise RuntimeError("piper_train is not installed in the current Python environment")
PIPER_TRAIN_DIR = pathlib.Path(_spec.submodule_search_locations[0])

init_py = PIPER_TRAIN_DIR / "vits" / "monotonic_align" / "__init__.py"
text = init_py.read_text(encoding="utf-8")

OLD = "    maximum_path_c(path, neg_cent, t_t_max, t_s_max)\n"
NEW = (
    "    assert t_t_max.max() <= path.shape[1] and t_s_max.max() <= path.shape[2], (\n"
    "        f\"monotonic_align bounds diagnostic: t_t_max={t_t_max.max()} vs \"\n"
    "        f\"path.shape[1]={path.shape[1]}, t_s_max={t_s_max.max()} vs \"\n"
    "        f\"path.shape[2]={path.shape[2]}\"\n"
    "    )\n"
    "    maximum_path_c(path, neg_cent, t_t_max, t_s_max)\n"
)

if "monotonic_align bounds diagnostic" in text:
    print("monotonic_align bounds-check diagnostic: already applied")
elif OLD not in text:
    raise RuntimeError(
        f"Could not find expected maximum_path_c call in {init_py} — "
        "piper-train's monotonic_align/__init__.py may have changed shape."
    )
else:
    text = text.replace(OLD, NEW, 1)
    init_py.write_text(text, encoding="utf-8")
    print(f"monotonic_align bounds-check diagnostic: applied to {init_py}")

cache = init_py.parent / "__pycache__"
if cache.exists():
    for pyc in cache.glob("__init__.cpython-*.pyc"):
        pyc.unlink()
