"""
Adds a NaN/Inf check on neg_cent right before it's used by maximum_path(),
in piper_train/vits/monotonic_align/__init__.py.

Both bounds diagnostics (upper AND lower, i.e. zero-length utterances) have
now been ruled out -- t_t_max/t_s_max are always within [1, shape] for every
sample, yet maximum_path_c still segfaults identically on GPU and CPU with
threading removed.

Re-reading maximum_path_each's backward pass in core.pyx:

    for y in range(t_y - 1, -1, -1):
        path[y, index] = 1
        if index != 0 and (index == y or value[y-1, index] < value[y-1, index-1]):
            index = index - 1

For a *correct* monotonic alignment, index is guaranteed to reach 0 by the
time y reaches 0 (every valid path starts at (0, 0)), so `index != 0` is
false at y=0 and the `value[y-1, ...]` accesses inside the `or` are never
evaluated. But if `value` (== neg_cent, updated in-place as the DP score
accumulates) ever contains NaN or Inf, every comparison against a NaN is
False under IEEE 754, so `index` can fail to reach 0 -- and at y=0 that
evaluates `value[-1, index]`, i.e. value[y-1, ...] with y=0. core.pyx has
@cython.wraparound(False), so a -1 index isn't Python-wrapped to the last row;
it computes a raw pointer one row before the buffer. That's the access
violation, and it's inherent to bad *floating-point content* in neg_cent, not
its shape -- which is exactly the theory the two bounds checks couldn't catch.

This adds a finite-value assertion on neg_cent right before the native call,
turning a theorized NaN/Inf-driven native crash into a normal, catchable
AssertionError -- diagnostic only; if it fires, the real fix is upstream,
wherever neg_cent (the alignment log-likelihood matrix) is computed in
models.py, not here.

Idempotent -- safe to run every setup.

Run after installing piper-train:
    <piper_venv>\\Scripts\\python scripts\\patch_piper_train_monotonic_align_nan_check.py
"""
import importlib.util
import pathlib

_spec = importlib.util.find_spec("piper_train")
if _spec is None:
    raise RuntimeError("piper_train is not installed in the current Python environment")
PIPER_TRAIN_DIR = pathlib.Path(_spec.submodule_search_locations[0])

init_py = PIPER_TRAIN_DIR / "vits" / "monotonic_align" / "__init__.py"
text = init_py.read_text(encoding="utf-8")

OLD = "    neg_cent = neg_cent.data.cpu().numpy().astype(np.float32)\n"
NEW = (
    "    neg_cent = neg_cent.data.cpu().numpy().astype(np.float32)\n"
    "    assert np.isfinite(neg_cent).all(), (\n"
    "        f\"monotonic_align nan diagnostic: neg_cent has \"\n"
    "        f\"{np.isnan(neg_cent).sum()} NaN and {np.isinf(neg_cent).sum()} \"\n"
    "        f\"Inf values out of {neg_cent.size}\"\n"
    "    )\n"
)

if "monotonic_align nan diagnostic" in text:
    print("monotonic_align NaN/Inf diagnostic: already applied")
elif OLD not in text:
    raise RuntimeError(
        f"Could not find expected neg_cent assignment in {init_py} — "
        "piper-train's monotonic_align/__init__.py may have changed shape."
    )
else:
    text = text.replace(OLD, NEW, 1)
    init_py.write_text(text, encoding="utf-8")
    print(f"monotonic_align NaN/Inf diagnostic: applied to {init_py}")

cache = init_py.parent / "__pycache__"
if cache.exists():
    for pyc in cache.glob("__init__.cpython-*.pyc"):
        pyc.unlink()
