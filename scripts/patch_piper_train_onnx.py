"""
Patches piper_train/export_onnx.py to explicitly request the legacy
TorchScript-based ONNX exporter (dynamo=False).

PyTorch 2.9 flipped torch.onnx.export's default from dynamo=False to
dynamo=True. The new dynamo-based path requires the optional `onnxscript`
package (not installed, not declared in piper-train's or our own
requirements) and is a completely different, newer export code path
piper's model was never written or tested against. Explicitly requesting
the old exporter restores the behavior this script actually expects.

Idempotent — safe to run every setup.

Run after installing piper-train:
    <piper_venv>\\Scripts\\python scripts\\patch_piper_train_onnx.py
"""
import importlib.util
import pathlib

_spec = importlib.util.find_spec("piper_train")
if _spec is None:
    raise RuntimeError("piper_train is not installed in the current Python environment")
PIPER_TRAIN_DIR = pathlib.Path(_spec.submodule_search_locations[0])

export_onnx = PIPER_TRAIN_DIR / "export_onnx.py"
text = export_onnx.read_text(encoding="utf-8")

OLD = "torch.onnx.export(\n        model=model_g,"
NEW = "torch.onnx.export(\n        dynamo=False,\n        model=model_g,"

if "dynamo=False" in text:
    print("dynamo=False (legacy ONNX exporter) patch: already applied")
elif OLD not in text:
    raise RuntimeError(
        f"Could not find expected torch.onnx.export(...) call in {export_onnx} — "
        "piper-train's export_onnx.py may have changed shape."
    )
else:
    text = text.replace(OLD, NEW, 1)
    export_onnx.write_text(text, encoding="utf-8")
    print(f"dynamo=False (legacy ONNX exporter) patch: applied to {export_onnx}")

cache = PIPER_TRAIN_DIR / "__pycache__"
if cache.exists():
    for pyc in cache.glob("export_onnx.cpython-*.pyc"):
        pyc.unlink()
