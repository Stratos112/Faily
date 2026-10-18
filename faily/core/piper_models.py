"""Per-character registry of trained Piper models.

Each saved model lives in outputs/characters/<name>/models/<id>/:

    model.onnx, model.onnx.json   what SPEAK / preview synthesis uses
    checkpoint.ckpt               the Lightning checkpoint it was exported from
                                  (~800 MB; needed to continue training it,
                                  can be deleted to free space)
    meta.json                     how it was made (base voice, epochs, learning
                                  rate, match options, clips, losses, ...)
    model.score.json              cached pronunciation score (modules/pronunciation.py)

The character's config.json "piper_model" points at the ACTIVE model's .onnx.
Everything under models/ survives retraining — training only wipes
piper_train/ and piper_dataset/.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
from pathlib import Path

from faily.core import characters as C

ROOT_NAME = "models"


def models_root(name: str) -> Path:
    return C.CHARACTERS_DIR / name / ROOT_NAME


def _meta_path(d: Path) -> Path:
    return d / "meta.json"


def _read(d: Path) -> dict | None:
    try:
        meta = json.loads(_meta_path(d).read_text())
    except (OSError, ValueError):
        return None
    onnx = d / "model.onnx"
    if not onnx.exists():
        return None
    ckpt = d / "checkpoint.ckpt"
    meta.update(
        id=d.name, dir=d, onnx=onnx,
        ckpt=ckpt if ckpt.exists() else None,
        ckpt_mb=round(ckpt.stat().st_size / 2**20) if ckpt.exists() else 0,
    )
    return meta


def list_models(name: str) -> list[dict]:
    """Saved models, newest first, each with .id .dir .onnx .ckpt (or None) and
    the meta.json fields. `active` marks the one the character speaks with."""
    root = models_root(name)
    out = [m for d in sorted(root.glob("*")) if d.is_dir() and (m := _read(d))] if root.is_dir() else []
    active = active_model_id(name, out)
    for m in out:
        m["active"] = m["id"] == active
    return sorted(out, key=lambda m: m.get("created", ""), reverse=True)


def get_model(name: str, model_id: str) -> dict | None:
    d = models_root(name) / model_id
    m = _read(d) if d.is_dir() else None
    if m:
        m["active"] = m["id"] == active_model_id(name)
    return m


def active_model_id(name: str, models: list[dict] | None = None) -> str | None:
    char = C.get_character(name) or {}
    current = char.get("piper_model") or ""
    if not current:
        return None
    try:
        cur = Path(current).resolve()
    except OSError:
        return None
    root = models_root(name)
    for d in root.glob("*") if root.is_dir() else []:
        try:
            if (d / "model.onnx").resolve() == cur:
                return d.name
        except OSError:
            continue
    return None


def _next_id(name: str) -> str:
    root = models_root(name)
    nums = [int(m.group(1)) for d in (root.glob("m*") if root.is_dir() else []) if (m := re.fullmatch(r"m(\d+)", d.name))]
    return f"m{(max(nums) + 1) if nums else 1:03d}"


def next_default_name(name: str) -> str:
    return f"Model {len(list_models(name)) + 1}"


def save_model(name: str, *, onnx: Path, ckpt: Path | None, meta: dict,
               activate: bool = True, replace_id: str | None = None) -> dict:
    """Copy a trained model (and its checkpoint) into the registry.
    replace_id: delete that model afterwards and let the new one take its place
    (name carried over unless meta already has one). Returns the saved entry."""
    old = get_model(name, replace_id) if replace_id else None
    mid = _next_id(name)
    d = models_root(name) / mid
    d.mkdir(parents=True)
    try:
        shutil.copy2(onnx, d / "model.onnx")
        cfg = onnx.with_suffix(".onnx.json")
        if cfg.exists():
            shutil.copy2(cfg, d / "model.onnx.json")
        score = onnx.with_suffix(".score.json")
        if score.exists():
            shutil.copy2(score, d / "model.score.json")
        if ckpt and ckpt.exists():
            shutil.copy2(ckpt, d / "checkpoint.ckpt")
        meta = {**meta, "created": meta.get("created") or time.strftime("%Y-%m-%dT%H:%M:%S")}
        if old and not meta.get("name"):
            meta["name"] = old.get("name")
        if old and "parent" not in meta:
            meta["parent"] = old.get("parent")
        meta.setdefault("name", next_default_name(name))
        _meta_path(d).write_text(json.dumps(meta, indent=2, default=str))
    except Exception:
        shutil.rmtree(d, ignore_errors=True)
        raise
    was_active = bool(old and old.get("active"))
    if old:
        delete_model(name, replace_id, _reassign=False)
    if activate or was_active:
        set_active(name, mid)
    return get_model(name, mid)


def update_meta(name: str, model_id: str, **fields) -> None:
    d = models_root(name) / model_id
    meta = json.loads(_meta_path(d).read_text())
    meta.update(fields)
    _meta_path(d).write_text(json.dumps(meta, indent=2, default=str))


def set_active(name: str, model_id: str) -> None:
    m = get_model(name, model_id)
    if not m:
        raise FileNotFoundError(f"No model '{model_id}' for '{name}'")
    C.set_piper_model(name, str(m["onnx"]))


def delete_model(name: str, model_id: str, _reassign: bool = True) -> None:
    """Remove a model. If it was the active one, the newest remaining model
    becomes active (or the character is left with no Piper model)."""
    d = models_root(name) / model_id
    was_active = active_model_id(name) == model_id
    shutil.rmtree(d, ignore_errors=True)
    if was_active and _reassign:
        rest = list_models(name)
        C.set_piper_model(name, str(rest[0]["onnx"]) if rest else "")


def delete_checkpoint(name: str, model_id: str) -> None:
    """Free the ~800 MB checkpoint; the model still speaks but can't be continued."""
    (models_root(name) / model_id / "checkpoint.ckpt").unlink(missing_ok=True)


# ── adopting a model trained before the registry existed ─────────────────────

def _ckpt_epoch(py: str, ckpt: Path) -> int | None:
    code = (
        "import torch\n"
        f"print(torch.load({str(ckpt)!r}, map_location='cpu', weights_only=False).get('epoch', 0))\n"
    )
    try:
        r = subprocess.run([py, "-c", code], capture_output=True, text=True, timeout=120)
        return int(r.stdout.strip().splitlines()[-1])
    except Exception:
        return None


def ensure_migrated(name: str) -> None:
    """If the character has a piper model from before the registry (config's
    piper_model points outside models/), adopt it as the first entry. Its
    checkpoint is recovered from piper_train/ when one was saved at the time the
    model was exported (matched by file time); fields that can't be known are
    listed in meta["inferred"] / left out."""
    char = C.get_character(name)
    if not char:
        return
    current = char.get("piper_model") or ""
    if not current or active_model_id(name) is not None:
        return
    src = Path(current)
    if not src.exists():
        return
    # Lazy imports: modules.piper pulls in heavier deps than core should.
    from faily.modules.piper import BASE_VOICES, _python, base_voice_paths, base_voice_ready, loss_at_epoch

    char_dir = C.CHARACTERS_DIR / name
    ckpt, epoch, inferred_ckpt = None, None, False
    t_onnx = src.stat().st_mtime
    cands = [p for p in (char_dir / "piper_train").rglob("*.ckpt")
             if t_onnx - 1800 <= p.stat().st_mtime <= t_onnx + 60]
    if cands:
        ckpt = max(cands, key=lambda p: p.stat().st_mtime)
        inferred_ckpt = True
        m = re.search(r"epoch=(\d+)", ckpt.name)
        epoch = int(m.group(1)) if m else _ckpt_epoch(str(_python()), ckpt)
    inferred = ["checkpoint"] if inferred_ckpt else []
    base_voice, base_epoch = None, None
    if epoch is not None:
        best = None
        for key in BASE_VOICES:
            if not base_voice_ready(key):
                continue
            be = _ckpt_epoch(str(_python()), base_voice_paths(key)[0])
            if be is not None and be <= epoch and (best is None or be > best[1]):
                best = (key, be)
        if best:
            base_voice, base_epoch = best
            inferred += ["base_voice", "base_epoch"]
    meta = {
        "name": "Imported model",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t_onnx)),
        "base_voice": base_voice,
        "base_voice_label": BASE_VOICES.get(base_voice, {}).get("label", base_voice) if base_voice else None,
        "base_epoch": base_epoch,
        "epoch": epoch,
        "epochs_trained": (epoch - base_epoch) if (epoch is not None and base_epoch is not None) else None,
        "imported": True,
        "inferred": inferred,
        "note": "Trained before models were tracked, so its settings (learning rate, match options, "
                "clips) weren't recorded."
                + (" Its checkpoint was matched by file time and is the best guess of where it stopped."
                   if ckpt else " Its checkpoint wasn't kept, so it can't be continued."),
    }
    if epoch is not None:
        try:
            meta["loss"] = loss_at_epoch(char_dir, epoch)
        except Exception:
            pass
    save_model(name, onnx=src, ckpt=ckpt, meta=meta, activate=True)
