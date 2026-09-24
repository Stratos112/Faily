"""
Adds a bounds assertion to piper_train/vits/monotonic_align/__init__.py's
maximum_path(), right before it calls into the native maximum_path_c extension.

Training crashes on the very first batch with a silent native access
violation (exit 0xC0000005, zero Python traceback) inside maximum_path_c,
called from maximum_path() (see faulthandler output: models.py:647 ->
monotonic_align/__init__.py). The crash reproduces identically on GPU and
CPU and with OpenMP parallelism removed, ruling out CUDA/driver corruption
and threading — it's a pure, deterministic bug reachable from CPU-only code.

v1 of this diagnostic only checked the upper bound (t_max <= path.shape),
which passed — but maximum_path_each computes `index = t_x - 1`, and if any
single utterance in the batch has t_x == 0 (a zero-length sequence along that
axis — plausible since t_t_max/t_s_max are derived via mask.sum(...)[:, 0],
which can silently come out 0 if the *other* axis is also degenerate for that
utterance), index becomes -1. core.pyx has @cython.wraparound(False) on both
functions, which disables Python-style negative-index wraparound at the C
level, so path[y, -1] doesn't wrap to the last column — it writes to raw
memory one slot before the buffer. That's an access violation, deterministic
and platform-independent, exactly matching what we've observed. v2 adds a
lower-bound (>= 1) check to catch this and report the exact offending shapes.

Re-enabling bounds checking inside the compiled extension itself runs into
Cython's nogil/void exception-propagation rules (maximum_path_each returns
void with no `except *` clause, so an exception raised during a nogil bounds
check isn't guaranteed to propagate cleanly). Asserting in the pure-Python
wrapper, before the native call, sidesteps all of that — diagnostic only;
remove once the real bug (likely a zero-length clip in the dataset) is fixed
at the source.

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

FRESH_OLD = "    maximum_path_c(path, neg_cent, t_t_max, t_s_max)\n"
V1_BLOCK = (
    "    assert t_t_max.max() <= path.shape[1] and t_s_max.max() <= path.shape[2], (\n"
    "        f\"monotonic_align bounds diagnostic: t_t_max={t_t_max.max()} vs \"\n"
    "        f\"path.shape[1]={path.shape[1]}, t_s_max={t_s_max.max()} vs \"\n"
    "        f\"path.shape[2]={path.shape[2]}\"\n"
    "    )\n"
    "    maximum_path_c(path, neg_cent, t_t_max, t_s_max)\n"
)
V2_BLOCK = (
    "    assert t_t_max.max() <= path.shape[1] and t_s_max.max() <= path.shape[2], (\n"
    "        f\"monotonic_align bounds diagnostic: t_t_max={t_t_max.max()} vs \"\n"
    "        f\"path.shape[1]={path.shape[1]}, t_s_max={t_s_max.max()} vs \"\n"
    "        f\"path.shape[2]={path.shape[2]}\"\n"
    "    )\n"
    "    assert t_t_max.min() >= 1 and t_s_max.min() >= 1, (\n"
    "        f\"monotonic_align zero-length diagnostic: t_t_max={t_t_max.tolist()}, \"\n"
    "        f\"t_s_max={t_s_max.tolist()} — a zero here means index=t_x-1 goes\"\n"
    "        f\" negative, and wraparound(False) turns path[y,-1] into an\"\n"
    "        f\" out-of-bounds write instead of a Python-level error\"\n"
    "    )\n"
    "    maximum_path_c(path, neg_cent, t_t_max, t_s_max)\n"
)

if "monotonic_align zero-length diagnostic" in text:
    print("monotonic_align bounds-check diagnostic: already at v2 (upper + lower bound)")
elif "monotonic_align bounds diagnostic" in text:
    if V1_BLOCK not in text:
        raise RuntimeError(
            f"Found v1 marker but not the expected v1 block text in {init_py} — "
            "it may have been hand-edited since."
        )
    text = text.replace(V1_BLOCK, V2_BLOCK, 1)
    init_py.write_text(text, encoding="utf-8")
    print(f"monotonic_align bounds-check diagnostic: upgraded v1 -> v2 (added zero-length check) in {init_py}")
elif FRESH_OLD not in text:
    raise RuntimeError(
        f"Could not find expected maximum_path_c call in {init_py} — "
        "piper-train's monotonic_align/__init__.py may have changed shape."
    )
else:
    text = text.replace(FRESH_OLD, V2_BLOCK, 1)
    init_py.write_text(text, encoding="utf-8")
    print(f"monotonic_align bounds-check diagnostic: applied (v2) to {init_py}")

cache = init_py.parent / "__pycache__"
if cache.exists():
    for pyc in cache.glob("__init__.cpython-*.pyc"):
        pyc.unlink()
