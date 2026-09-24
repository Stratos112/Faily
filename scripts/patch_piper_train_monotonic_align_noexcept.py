"""
Declares maximum_path_each and maximum_path_c as `noexcept` in piper_train's
monotonic_align core.pyx, then rebuilds the extension.

Every prior diagnostic on the Python side has come back clean: t_t_max/t_s_max
are always within [1, shape] (upper AND lower bound checked), neg_cent is
always finite, the crash reproduces identically on GPU and CPU, and removing
the prange/OpenMP parallelism (patch_piper_train_monotonic_align_serial.py)
didn't change anything. That rules out every theory about the *data* being
malformed, and rules out GPU/CUDA and batch-level threading as the cause.

The rebuild logs, however, contained a compiler diagnostic we'd been treating
as noise:

    performance hint: core.pyx:5:0: Exception check on 'maximum_path_each'
    will always require the GIL to be acquired.
    Possible solutions:
      1. Declare 'maximum_path_each' as 'noexcept' if you control the
         definition and you're sure you don't want the function to raise
         exceptions.

Neither maximum_path_each nor maximum_path_c is declared `noexcept`, so even
though both are `nogil`, Cython still emits an implicit exception-state check
after every single call — which means acquiring the GIL, checking it, and
releasing it again, once per batch item, regardless of whether the loop is
prange or a plain serial range (this is generated at the call site, not by
the loop construct, which is exactly why removing prange didn't help). A
`nogil` function secretly re-acquiring the GIL on every call, in a process
that also has a background TensorBoard writer thread doing its own
GIL-releasing `queue.get()`, is a real place for a low-level interpreter
thread-state race to live on Windows — and it would reproduce identically on
GPU/CPU and with/without prange, exactly as observed.

Neither function actually raises a Python exception in practice, so
`noexcept` is both correct and removes the implicit GIL dance entirely.

Idempotent -- safe to run every setup. Rebuilds the extension in place.

Run after installing piper-train:
    <piper_venv>\\Scripts\\python scripts\\patch_piper_train_monotonic_align_noexcept.py
"""
import importlib.util
import pathlib
import subprocess
import sys

_spec = importlib.util.find_spec("piper_train")
if _spec is None:
    raise RuntimeError("piper_train is not installed in the current Python environment")
PIPER_TRAIN_DIR = pathlib.Path(_spec.submodule_search_locations[0])
MONO_DIR = PIPER_TRAIN_DIR / "vits" / "monotonic_align"
core_pyx = MONO_DIR / "core.pyx"

text = core_pyx.read_text(encoding="utf-8")

# The serial-loop patch (patch_piper_train_monotonic_align_serial.py) changes
# `for i in prange(b, nogil=True):` to a plain `for i in range(b):` but
# doesn't touch either function signature line, so this replacement applies
# cleanly whether or not that patch has already run.
REPLACEMENTS = [
    (
        "cdef void maximum_path_each(int[:,::1] path, float[:,::1] value, int t_y, int t_x, float max_neg_val=-1e9) nogil:\n",
        "cdef void maximum_path_each(int[:,::1] path, float[:,::1] value, int t_y, int t_x, float max_neg_val=-1e9) noexcept nogil:\n",
    ),
    (
        "cpdef void maximum_path_c(int[:,:,::1] paths, float[:,:,::1] values, int[::1] t_ys, int[::1] t_xs) nogil:\n",
        "cpdef void maximum_path_c(int[:,:,::1] paths, float[:,:,::1] values, int[::1] t_ys, int[::1] t_xs) noexcept nogil:\n",
    ),
]

applied = False
for old, new in REPLACEMENTS:
    if new in text:
        continue
    if old not in text:
        raise RuntimeError(
            f"Could not find expected signature {old!r} in {core_pyx} — "
            "piper-train's core.pyx may have changed shape, or noexcept is "
            "already applied differently."
        )
    text = text.replace(old, new, 1)
    applied = True

if applied:
    core_pyx.write_text(text, encoding="utf-8")
    print(f"monotonic_align noexcept patch: applied to {core_pyx}")
else:
    print("monotonic_align noexcept patch: already applied")

print("Rebuilding monotonic_align extension...")
site_packages = PIPER_TRAIN_DIR.parent
result = subprocess.run(
    [sys.executable, str(MONO_DIR / "setup.py"), "build_ext", "--inplace"],
    cwd=str(site_packages),
)
if result.returncode != 0:
    raise RuntimeError("monotonic_align rebuild failed")
print("monotonic_align rebuilt with noexcept (no implicit GIL exception-check)")
