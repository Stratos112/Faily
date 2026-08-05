"""
Patches bigvgan.BigVGAN._from_pretrained so that proxies and resume_download
are no longer required keyword-only arguments.

The original signature already ends with **model_kwargs, so the fix is:
  - DELETE the proxies and resume_download lines from the signature
  - Extract them from model_kwargs at the top of the body

Run once after installing bigvgan:
    .venv\Scripts\python scripts\patch_bigvgan.py
"""
import re
import shutil
import pathlib
import importlib.util

_spec = importlib.util.find_spec("bigvgan")
if _spec is None:
    raise RuntimeError("bigvgan is not installed in the current Python environment")
BIGVGAN_PATH = pathlib.Path(_spec.origin)
BAK = BIGVGAN_PATH.with_suffix(".py.bak")

# ── Restore from backup if one exists ─────────────────────────────────────
if BAK.exists():
    shutil.copy(BAK, BIGVGAN_PATH)
    print(f"Restored from backup: {BAK}")

src = BIGVGAN_PATH.read_text(encoding="utf-8")

# ── Show current signature ─────────────────────────────────────────────────
idx = src.find("def _from_pretrained")
print("=== _from_pretrained signature (first 700 chars) ===")
print(src[idx : idx + 700])
print()

# ── 1. Delete proxies / resume_download lines from signature ──────────────
# Matches both "required" form and "= default" form, with any whitespace.
before = src
src = re.sub(
    r"[ \t]+proxies\s*:[^,\n]+,\s*\n[ \t]+resume_download\s*:[^,\n]+,\s*\n",
    "",   # just delete — **model_kwargs already at end absorbs them if passed
    src,
)
if src == before:
    print("WARNING: proxies/resume_download pattern not found — printing sig for inspection:")
    print(src[idx : idx + 700])
else:
    print("✓  Deleted proxies / resume_download from signature.")

# ── 2. Inject body extractions from model_kwargs ──────────────────────────
INJECTION = (
    '        proxies = model_kwargs.pop("proxies", None)\n'
    '        resume_download = model_kwargs.pop("resume_download", False)\n'
)

m = re.search(r"def _from_pretrained\b.*?\):\n", src, re.DOTALL)
if m:
    body_start = m.end()
    # Skip docstring if present so we inject after it
    after = src[body_start:]
    doc_m = re.match(r'(\s+""".*?"""[ \t]*\n)', after, re.DOTALL)
    if doc_m:
        insert_at = body_start + doc_m.end()
    else:
        insert_at = body_start
    if INJECTION.strip() not in src[insert_at : insert_at + 300]:
        src = src[:insert_at] + INJECTION + src[insert_at:]
        print("✓  Injected model_kwargs.pop() extractions into body.")
    else:
        print("  Body extraction already present — skipping.")
else:
    print("WARNING: could not locate _from_pretrained body.")

# ── 3. Write ───────────────────────────────────────────────────────────────
shutil.copy(BIGVGAN_PATH, BAK)
BIGVGAN_PATH.write_text(src, encoding="utf-8")
print(f"✓  Written: {BIGVGAN_PATH}")
print(f"  Backup:  {BAK}")

# ── 4. Delete stale .pyc ───────────────────────────────────────────────────
cache = BIGVGAN_PATH.parent / "__pycache__"
for pyc in cache.glob("bigvgan.cpython-*.pyc"):
    pyc.unlink()
    print(f"✓  Deleted: {pyc.name}")

# ── 5. Show result ─────────────────────────────────────────────────────────
src2 = BIGVGAN_PATH.read_text(encoding="utf-8")
idx2 = src2.find("def _from_pretrained")
print("\n=== patched result (first 700 chars) ===")
print(src2[idx2 : idx2 + 700])
print("\nDone. Restart Faily and try Seed-VC.")
