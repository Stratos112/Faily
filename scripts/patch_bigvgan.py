"""
Patches bigvgan.BigVGAN._from_pretrained so that proxies and resume_download
are no longer required keyword-only arguments.

HF Hub >=0.24 stopped forwarding these to _from_pretrained, but bigvgan still
declares them as required. Python 3.14's classmethod dispatch ignores
__kwdefaults__ in some specialised call paths, so adding defaults to the
source is not enough — we have to remove the parameters from the signature
entirely and absorb them via **kwargs instead.

Run once from the Windows command line:
    C:\Python314\python.exe scripts\patch_bigvgan.py
"""
import re
import shutil
import pathlib

BIGVGAN_PATH = pathlib.Path(
    r"C:\Users\Sky\AppData\Roaming\Python\Python314\site-packages\bigvgan\bigvgan.py"
)

if not BIGVGAN_PATH.exists():
    raise FileNotFoundError(f"bigvgan.py not found at {BIGVGAN_PATH}")

src = BIGVGAN_PATH.read_text(encoding="utf-8")

# ── 1. Show current _from_pretrained signature ─────────────────────────────
idx = src.find("def _from_pretrained")
print("=== current _from_pretrained (first 600 chars) ===")
print(src[idx : idx + 600])
print()

# ── 2. Remove proxies / resume_download from signature ────────────────────
# Handles both the original (no default) and our previous patch (with defaults).
before = src
src = re.sub(
    r"[ \t]+proxies\s*:[^,\n]+,\s*\n[ \t]+resume_download\s*:[^,\n]+,",
    "\n        **kwargs,",
    src,
)
if src == before:
    print("WARNING: signature pattern not found — file may already be patched or differ.")
    print("Printing _from_pretrained for manual inspection:")
    print(src[idx : idx + 800])
else:
    print("✓  Removed proxies / resume_download from signature, added **kwargs.")

# ── 3. Inject body variable extraction after the closing ): ───────────────
INJECTION = (
    "        proxies = kwargs.pop(\"proxies\", None)\n"
    "        resume_download = kwargs.pop(\"resume_download\", False)\n"
)

# Find the method's body start: after the first ): that closes _from_pretrained
m = re.search(r"def _from_pretrained\b(.*?)\):\n", src, re.DOTALL)
if m:
    body_start = m.end()
    if INJECTION.strip() not in src[body_start : body_start + 300]:
        src = src[:body_start] + INJECTION + src[body_start:]
        print("✓  Injected proxies/resume_download extraction into body.")
    else:
        print("  Body extraction already present — skipping.")
else:
    print("WARNING: could not locate _from_pretrained body — no injection done.")

# ── 4. Write patched file ──────────────────────────────────────────────────
bak = BIGVGAN_PATH.with_suffix(".py.bak")
shutil.copy(BIGVGAN_PATH, bak)
print(f"  Backup: {bak}")

BIGVGAN_PATH.write_text(src, encoding="utf-8")
print(f"✓  Written: {BIGVGAN_PATH}")

# ── 5. Delete stale .pyc so Python recompiles from patched source ──────────
cache = BIGVGAN_PATH.parent / "__pycache__"
deleted = []
for pyc in cache.glob("bigvgan.cpython-*.pyc"):
    pyc.unlink()
    deleted.append(pyc.name)
if deleted:
    print(f"✓  Deleted stale .pyc: {', '.join(deleted)}")

print("\nDone. Restart Faily and try Seed-VC again.")
