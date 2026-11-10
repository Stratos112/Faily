"""What a character is, what it can do, and where it needs work — the data behind the ABOUT tab.
Pure logic over the character's files; no UI, no models loaded."""
from faily.core import characters as C
from faily.core import piper_models as PM
from faily.core import refs

MIN_RECOGNIZABLE_MIN = 10.0        # minutes of active audio for a recognizable fine-tuned voice
MIN_STRONG_MIN = 30.0              # ...for a strong one
MAX_INFLATED_SHARE = 0.75          # above this share of active audio being generated, add real recordings
MIN_TRANSCRIPT_SHARE = 0.95
RETRAIN_CHANGE_SHARE = 0.10        # active clip count moved by this much since the last training
GOOD_PRONUNCIATION = 0.90
OK_PRONUNCIATION = 0.75


def _rec(level: str, text: str, action: str | None = None) -> dict:
    return {"level": level, "text": text, "action": action}                   # level: warn | todo | info | ok


def _minutes(seconds: float) -> float:
    return seconds / 60.0


def _model_summary(name: str) -> dict | None:
    models = PM.list_models(name)
    active = next((m for m in models if m.get("active")), None)
    if not active:
        return None
    score = None
    try:
        from faily.modules.pronunciation import cached_score
        score = cached_score(active["onnx"])
    except Exception:                                                   # noqa: BLE001  a missing score is not an error
        pass
    clips = active.get("clips") or {}
    return {
        "id": active["id"], "name": active.get("name", active["id"]), "created": active.get("created", ""),
        "epoch": active.get("epoch"), "base_voice": active.get("base_voice_label") or active.get("base_voice", ""),
        "clips": clips.get("n"), "minutes": clips.get("minutes"),
        "accuracy": (score or {}).get("accuracy"), "words_per_sec": (score or {}).get("words_per_sec"),
        "trained_seconds": active.get("trained_seconds"), "count": len(models),
    }


def _details(rows: list[dict], inherited: list[dict], stats: dict) -> dict:
    """General facts about the character's audio, for the ABOUT page."""
    secs = sorted(r["seconds"] for r in rows if r["seconds"] > 0)
    median = secs[len(secs) // 2] if secs else 0.0
    return {
        "clips": stats["all"]["clips"], "minutes": _minutes(stats["all"]["seconds"]),
        "active_clips": stats["active"]["clips"], "active_minutes": _minutes(stats["active"]["seconds"]),
        "median_clip": median, "longest_clip": secs[-1] if secs else 0.0,
        "transcript_share": stats["transcript_share"],
        "no_cloning": sum(1 for r in rows if not r["clone"]),
        "inflated_by_backend": {k: v["clips"] for k, v in stats["by_backend"].items()},
        "inflated_by_category": {k: v["clips"] for k, v in stats["by_category"].items()},
        "inherited_clips": len(inherited),
    }


def character_profile(name: str) -> dict | None:
    char = C.get_character(name)
    if not char:
        return None
    is_base = "parent" not in char
    rows = refs.clip_rows(name)
    inherited = refs.inherited_rows(name)
    stats = refs.summarize(rows)
    variants = [c["name"] for c in C.list_characters() if c.get("parent") == name]
    model = _model_summary(name)

    act_min = _minutes(stats["active"]["seconds"])
    act_inf_share = (stats["active_inflated"]["seconds"] / stats["active"]["seconds"]) if stats["active"]["seconds"] else 0.0

    # ── recommendations, most pressing first ─────────────────────────────────
    recs: list[dict] = []
    if not rows and not inherited:
        recs.append(_rec("warn", "No reference audio yet — add a recording to give this character a voice.", "references"))
    if rows and act_min < MIN_RECOGNIZABLE_MIN:
        recs.append(_rec("warn", f"Only {act_min:.1f} min of audio is active for training; a recognizable voice usually "
                                 f"needs {MIN_RECOGNIZABLE_MIN:.0f}+ min. Add recordings or switch more clips on.", "references"))
    elif rows and act_min < MIN_STRONG_MIN:
        recs.append(_rec("info", f"{act_min:.1f} min active — enough for a recognizable voice; {MIN_STRONG_MIN:.0f}+ min "
                                 "gives a stronger resemblance.", "references"))
    if act_inf_share > MAX_INFLATED_SHARE and stats["active_recordings"]["clips"]:
        recs.append(_rec("warn", f"{act_inf_share:.0%} of the active audio is generated. The model can't beat the clone it "
                                 "learns from — add more real recordings or lower the inflated mix on the TRAINING tab.", "training"))
    if rows and stats["transcript_share"] < MIN_TRANSCRIPT_SHARE:
        missing = len(rows) - stats["transcribed"]
        recs.append(_rec("warn", f"{missing} clip{'s' if missing != 1 else ''} have no transcript. Piper skips them when "
                                 "training; Qualify can suggest transcripts with Whisper.", "qualify"))
    if stats["long"]:
        recs.append(_rec("todo", f"{stats['long']} clip{'s' if stats['long'] != 1 else ''} are over {stats['long_threshold']:.0f} s. "
                                 "Very long clips risk out-of-memory crashes in training, so they are left out of runs — "
                                 "split them.", "references"))
    inactive = stats["all"]["clips"] - stats["active"]["clips"]
    if inactive:
        recs.append(_rec("info", f"{inactive} clip{'s' if inactive != 1 else ''} switched off for training "
                                 f"({_minutes(stats['all']['seconds'] - stats['active']['seconds']):.1f} min kept aside).", "references"))
    if rows and not model and act_min >= 3:
        recs.append(_rec("todo", "Data is ready but no voice has been trained yet.", "training"))
    if model and model["clips"]:
        moved = abs(stats["active"]["clips"] - model["clips"]) / max(model["clips"], 1)
        if moved >= RETRAIN_CHANGE_SHARE:
            recs.append(_rec("todo", f"The active set has {stats['active']['clips']} clips; the current voice was trained on "
                                     f"{model['clips']}. Retrain to use the changes.", "training"))
    if model and model["accuracy"] is not None and model["accuracy"] < OK_PRONUNCIATION:
        recs.append(_rec("warn", f"Pronunciation is weak ({model['accuracy']:.0%} of test words recognised). Try matching "
                                 "Pronunciation to the Base Model, or train on cleaner, correctly transcribed clips.", "training"))
    if model and model["accuracy"] is None:
        recs.append(_rec("info", "The current voice hasn't been scored yet — audition it on the MODELS tab.", None))
    if is_base and not variants and model:
        recs.append(_rec("info", "Create expression variants (calm, angry, …) from the TUNE tab.", None))
    if not recs:
        recs.append(_rec("ok", "Nothing needs attention.", None))
    order = {"warn": 0, "todo": 1, "info": 2, "ok": 3}
    recs.sort(key=lambda r: order[r["level"]])

    return {
        "name": name, "is_base": is_base, "parent": char.get("parent"), "created": char.get("created", ""),
        "variants": variants, "stats": stats, "inherited": {"clips": len(inherited),
                                                             "seconds": sum(r["seconds"] for r in inherited)},
        "model": model, "recommendations": recs, "details": _details(rows, inherited, stats),
        "favorites": len(C.list_character_favorites(name)) if is_base else 0,
        "history": sum(1 for _ in (C.CHARACTERS_DIR / name / "history").glob("*.wav")) if (C.CHARACTERS_DIR / name / "history").is_dir() else 0,
    }
