"""
Patches pytorch-lightning 1.7.x (pinned for piper-train) for compatibility
with a modern NumPy/PyTorch stack. Idempotent — safe to run every setup.

  1. np.Inf was removed in NumPy 2.0; pytorch-lightning 1.7.x still uses it
     in a couple of callbacks (ModelCheckpoint, EarlyStopping).
  2. PyTorch 2.6 flipped torch.load's default from weights_only=False to
     True; pytorch-lightning 1.7.x's checkpoint loader (cloud_io.py) never
     passed it explicitly, so loading any checkpoint with non-tensor
     objects pickled in (e.g. a pathlib.PosixPath) is blocked.
  3. The base checkpoint was saved on Linux and has a pickled
     pathlib.PosixPath baked in, which Windows refuses to instantiate
     natively ("cannot instantiate 'PosixPath' on your system") — alias it
     to WindowsPath for the duration of the load.

Run after installing pytorch-lightning:
    <piper_venv>\\Scripts\\python scripts\\patch_pytorch_lightning.py
"""
import importlib.util
import pathlib

_spec = importlib.util.find_spec("pytorch_lightning")
if _spec is None:
    raise RuntimeError("pytorch_lightning is not installed in the current Python environment")
PL_DIR = pathlib.Path(_spec.submodule_search_locations[0])


def _patch_file(path: pathlib.Path, replacements: list[tuple[str, str]]) -> bool:
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8")
    changed = False
    for old, new in replacements:
        if old in text:
            text = text.replace(old, new)
            changed = True
    if changed:
        path.write_text(text, encoding="utf-8")
    return changed


# ── 1. np.Inf → np.inf (NumPy 2.x compat) ────────────────────────────────────
n = 0
for f in PL_DIR.rglob("*.py"):
    if _patch_file(f, [("np.Inf", "np.inf")]):
        n += 1
print(f"NumPy 2.x compat (np.Inf -> np.inf): patched {n} file(s)")

# ── 2/3. cloud_io.py — weights_only + cross-platform PosixPath ──────────────
cloud_io = PL_DIR / "utilities" / "cloud_io.py"
text = cloud_io.read_text(encoding="utf-8")

if "_cross_platform_torch_load" in text:
    print("torch 2.6+ / cross-platform checkpoint compat: already patched")
else:
    # Normalize whatever state the two call sites are in (pristine, or
    # already carrying an earlier weights_only=False-only patch) down to a
    # single call to the new helper.
    for old in (
        "torch.load(path_or_url, map_location=map_location, weights_only=False)",
        "torch.load(path_or_url, map_location=map_location)",
    ):
        if old in text:
            text = text.replace(old, "_cross_platform_torch_load(path_or_url, map_location=map_location)")
            break
    for old in (
        "torch.load(f, map_location=map_location, weights_only=False)",
        "torch.load(f, map_location=map_location)",
    ):
        if old in text:
            text = text.replace(old, "_cross_platform_torch_load(f, map_location=map_location)")
            break

    if "_cross_platform_torch_load(" not in text:
        raise RuntimeError(
            f"Could not find expected torch.load(...) call sites in {cloud_io} — "
            "pytorch-lightning's cloud_io.py may have changed shape."
        )

    IMPORT_OLD = "import io\nfrom pathlib import Path"
    IMPORT_NEW = "import io\nimport pathlib as _pathlib_module\nimport sys\nfrom pathlib import Path"
    if IMPORT_OLD not in text:
        raise RuntimeError(f"Could not find expected imports in {cloud_io}")
    text = text.replace(IMPORT_OLD, IMPORT_NEW, 1)

    HELPER = (
        "def _cross_platform_torch_load(*args, **kwargs):\n"
        '    """Windows can\'t natively instantiate a pickled PosixPath (e.g. from a\n'
        "    checkpoint saved on Linux) -- temporarily alias it to WindowsPath so\n"
        "    unpickling succeeds. weights_only=False restores pre-2.6 torch.load\n"
        "    behavior, needed for Lightning checkpoints (trusted, our own download).\n"
        '    """\n'
        "    kwargs.setdefault(\"weights_only\", False)\n"
        "    if sys.platform != \"win32\":\n"
        "        return torch.load(*args, **kwargs)\n"
        "    _posix = _pathlib_module.PosixPath\n"
        "    _pathlib_module.PosixPath = _pathlib_module.WindowsPath\n"
        "    try:\n"
        "        return torch.load(*args, **kwargs)\n"
        "    finally:\n"
        "        _pathlib_module.PosixPath = _posix\n"
        "\n\n"
        "def load(\n"
    )
    if "\ndef load(\n" not in text:
        raise RuntimeError(f"Could not find 'def load(' in {cloud_io}")
    text = text.replace("def load(\n", HELPER, 1)

    cloud_io.write_text(text, encoding="utf-8")
    print(f"torch 2.6+ weights_only / cross-platform checkpoint compat: patched {cloud_io}")

# ── Delete stale __pycache__ so the patched source is actually used ─────────
cache = PL_DIR / "utilities" / "__pycache__"
if cache.exists():
    for pyc in cache.glob("cloud_io.cpython-*.pyc"):
        pyc.unlink()
