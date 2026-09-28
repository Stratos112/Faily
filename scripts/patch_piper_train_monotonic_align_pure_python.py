"""
Replaces piper_train/vits/monotonic_align's maximum_path() with a pure
Python/NumPy implementation, bypassing the compiled Cython extension
(monotonic_align.core.maximum_path_c) entirely.

Diagnostic history (see the other patch_piper_train_monotonic_align_*.py
scripts in this directory for full reasoning on each): training crashes with
a native access violation (exit 0xC0000005, zero Python traceback) inside
maximum_path_c on the very first training batch, 100% reproducibly. In order,
we ruled out:
  - an out-of-bounds t_t_max/t_s_max vs. path/neg_cent shape (upper bound)
  - a zero-length utterance driving index=-1 into a wraparound(False) index
    (lower bound)
  - NaN/Inf in neg_cent
  - cython.parallel.prange / OpenMP threading (crashes identically serial)
  - CUDA/GPU/driver corruption (crashes identically on --accelerator cpu)
  - missing `noexcept` causing an implicit per-call GIL exception-check
    (crashes identically after declaring both functions noexcept)

Every theory about the *input* being malformed, and every theory about *why*
the compiled extension might legitimately fault on valid input, checked out
clean. That leaves the extension itself broken at a level none of these
targeted fixes reach on this specific Windows/MSVC/Python 3.11 build — not
worth further guessing. This sidesteps the whole problem: same algorithm,
same interface, implemented in plain Python/NumPy instead of compiled Cython.
It's slower per training step (batch size is small, so this is a minor cost,
not a blocker), but it uses ordinary Python negative-indexing semantics, so
even the y=0/index!=0 edge case that motivated the noexcept theory just reads
the last row harmlessly instead of writing before the start of a buffer.

Idempotent — safe to run every setup. No rebuild needed (pure Python).

Run after installing piper-train:
    <piper_venv>\\Scripts\\python scripts\\patch_piper_train_monotonic_align_pure_python.py
"""
import importlib.util
import pathlib

_spec = importlib.util.find_spec("piper_train")
if _spec is None:
    raise RuntimeError("piper_train is not installed in the current Python environment")
PIPER_TRAIN_DIR = pathlib.Path(_spec.submodule_search_locations[0])

init_py = PIPER_TRAIN_DIR / "vits" / "monotonic_align" / "__init__.py"

MARKER = "pure Python/NumPy version"

NEW_CONTENT = '''import numpy as np
import torch


def maximum_path(neg_cent, mask):
    """Pure Python/NumPy version -- see
    patch_piper_train_monotonic_align_pure_python.py in the Faily repo for
    why the compiled Cython extension (monotonic_align.core.maximum_path_c)
    is bypassed here.
    """
    device = neg_cent.device
    dtype = neg_cent.dtype
    neg_cent = neg_cent.data.cpu().numpy().astype(np.float32)
    path = np.zeros(neg_cent.shape, dtype=np.int32)

    t_t_max = mask.sum(1)[:, 0].data.cpu().numpy().astype(np.int32)
    t_s_max = mask.sum(2)[:, 0].data.cpu().numpy().astype(np.int32)

    for i in range(neg_cent.shape[0]):
        _maximum_path_each(path[i], neg_cent[i], int(t_t_max[i]), int(t_s_max[i]))

    return torch.from_numpy(path).to(device=device, dtype=dtype)


def _maximum_path_each(path, value, t_y, t_x, max_neg_val=-1e9):
    index = t_x - 1

    for y in range(t_y):
        for x in range(max(0, t_x + y - t_y), min(t_x, y + 1)):
            if x == y:
                v_cur = max_neg_val
            else:
                v_cur = value[y - 1, x]
            if x == 0:
                v_prev = 0.0 if y == 0 else max_neg_val
            else:
                v_prev = value[y - 1, x - 1]
            value[y, x] += max(v_prev, v_cur)

    for y in range(t_y - 1, -1, -1):
        path[y, index] = 1
        if index != 0 and (index == y or value[y - 1, index] < value[y - 1, index - 1]):
            index -= 1
'''

text = init_py.read_text(encoding="utf-8")
if MARKER in text:
    print("monotonic_align pure-Python patch: already applied")
else:
    init_py.write_text(NEW_CONTENT, encoding="utf-8")
    print(f"monotonic_align pure-Python patch: applied to {init_py}")

cache = init_py.parent / "__pycache__"
if cache.exists():
    for pyc in cache.glob("__init__.cpython-*.pyc"):
        pyc.unlink()
