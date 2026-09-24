"""
Replaces the prange(nogil=True) OpenMP loop in piper_train's monotonic_align
core.pyx with a plain serial range() loop, then rebuilds the extension.

The bounds-check diagnostic (patch_piper_train_monotonic_align_bounds_check.py)
ruled out an out-of-bounds native index: t_t_max/t_s_max are within
path/neg_cent's bounds every time, yet maximum_path_c still segfaults on the
very first call. The remaining live suspect is the OpenMP parallelism itself
-- maximum_path_c's batch loop runs via cython.parallel.prange, a real
multi-threaded OpenMP kernel built by MSVC, invoked for the first time in a
process that has already initialized a CUDA context. A native thread pool
spawning for the first time and immediately corrupting memory in that
situation is a known-shaped failure on Windows (OpenMP runtime / CUDA driver
thread-local state interaction). Removing the parallelism entirely (serial
range() instead of prange) isolates whether threading is the actual cause,
independent of any data/shape theory.

This is a real behavior change, not just a diagnostic flag -- if it fixes the
crash, training simply runs on a single thread for this one alignment kernel
(it's a small DP loop per batch item; not a meaningful perf hit at batch
sizes this small). Safe to leave in place if it works.

Idempotent -- safe to run every setup. Rebuilds the extension in place.

Run after installing piper-train:
    <piper_venv>\\Scripts\\python scripts\\patch_piper_train_monotonic_align_serial.py
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

OLD = "  for i in prange(b, nogil=True):\n"
NEW = "  for i in range(b):  # prange disabled -- see patch_piper_train_monotonic_align_serial.py\n"

if "prange disabled" in text:
    print("monotonic_align serial-loop patch: already applied")
elif OLD not in text:
    raise RuntimeError(
        f"Could not find expected 'for i in prange(b, nogil=True):' in {core_pyx} — "
        "piper-train's core.pyx may have changed shape."
    )
else:
    text = text.replace(OLD, NEW, 1)
    core_pyx.write_text(text, encoding="utf-8")
    print(f"monotonic_align serial-loop patch: applied to {core_pyx}")

print("Rebuilding monotonic_align extension...")
site_packages = PIPER_TRAIN_DIR.parent
result = subprocess.run(
    [sys.executable, str(MONO_DIR / "setup.py"), "build_ext", "--inplace"],
    cwd=str(site_packages),
)
if result.returncode != 0:
    raise RuntimeError("monotonic_align rebuild failed")
print("monotonic_align rebuilt with prange disabled (serial loop)")
