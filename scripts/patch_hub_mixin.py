"""
Patches huggingface_hub/hub_mixin.py to explicitly pass proxies and
resume_download when calling cls._from_pretrained(), so that bigvgan's
implementation (which still requires them as keyword-only args) doesn't
raise TypeError.

Run once after installing bigvgan:
    .venv\Scripts\python scripts\patch_hub_mixin.py
"""
import re
import shutil
import pathlib
import importlib.util

_spec = importlib.util.find_spec("huggingface_hub")
if _spec is None:
    raise RuntimeError("huggingface_hub is not installed in the current Python environment")
HM_PATH = pathlib.Path(_spec.submodule_search_locations[0]) / "hub_mixin.py"
BAK = HM_PATH.with_suffix(".py.bak")

if not HM_PATH.exists():
    raise FileNotFoundError(f"hub_mixin.py not found at {HM_PATH}")

src = HM_PATH.read_text(encoding="utf-8")

# ── Show the current cls._from_pretrained call ────────────────────────────
idx = src.find("instance = cls._from_pretrained(")
if idx == -1:
    print("ERROR: could not find 'instance = cls._from_pretrained(' — printing around line 561:")
    lines = src.splitlines()
    for i, l in enumerate(lines[555:570], 556):
        print(f"{i}: {l}")
    raise SystemExit(1)

print("=== current cls._from_pretrained call ===")
print(src[idx : idx + 400])
print()

# ── Patch: inject proxies and resume_download before **model_kwargs ────────
OLD = "            **model_kwargs,\n        )"
NEW = (
    "            proxies=model_kwargs.pop(\"proxies\", None),\n"
    "            resume_download=model_kwargs.pop(\"resume_download\", False),\n"
    "            **model_kwargs,\n"
    "        )"
)

if NEW.strip() in src:
    print("Already patched — nothing to do.")
elif OLD not in src:
    # Try with different indentation
    for indent in ("    ", "      ", "        ", "          "):
        old2 = f"{indent}**model_kwargs,\n{indent[:-4]})"
        if old2 in src:
            new2 = (
                f"{indent}proxies=model_kwargs.pop(\"proxies\", None),\n"
                f"{indent}resume_download=model_kwargs.pop(\"resume_download\", False),\n"
                f"{indent}**model_kwargs,\n"
                f"{indent[:-4]})"
            )
            src = src.replace(old2, new2, 1)
            print(f"✓  Patched with indent={repr(indent)}")
            break
    else:
        print("ERROR: could not find **model_kwargs pattern. Printing the call area:")
        print(src[idx : idx + 400])
        raise SystemExit(1)
else:
    src = src.replace(OLD, NEW, 1)
    print("✓  Injected proxies / resume_download into cls._from_pretrained call.")

# ── Write ──────────────────────────────────────────────────────────────────
shutil.copy(HM_PATH, BAK)
HM_PATH.write_text(src, encoding="utf-8")
print(f"✓  Written: {HM_PATH}")
print(f"  Backup:  {BAK}")

# ── Delete stale .pyc ──────────────────────────────────────────────────────
cache = HM_PATH.parent / "__pycache__"
for pyc in cache.glob("hub_mixin.cpython-*.pyc"):
    pyc.unlink()
    print(f"✓  Deleted: {pyc.name}")

# ── Show result ────────────────────────────────────────────────────────────
src2 = HM_PATH.read_text(encoding="utf-8")
idx2 = src2.find("instance = cls._from_pretrained(")
print("\n=== patched call ===")
print(src2[idx2 : idx2 + 450])
print("\nDone. Restart Faily and try Seed-VC.")
