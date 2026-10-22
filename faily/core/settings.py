import json
import os
import shutil
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent.parent
# Local config/secrets live in their own dir so Docker can volume-mount it —
# the project root itself is baked into the image and lost on rebuild.
CONFIG_DIR = Path(os.environ.get("FAILY_CONFIG_DIR") or _PROJECT_ROOT / "config")


def config_file(name: str) -> Path:
    """Path to a file in CONFIG_DIR, moving it over from the project root
    first if an older install left it there."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    path = CONFIG_DIR / name
    legacy = _PROJECT_ROOT / name
    if not path.exists() and legacy.exists():
        shutil.move(str(legacy), str(path))
    return path


_SETTINGS_FILE = config_file("faily_settings.json")
_DEFAULTS: dict = {
    "default_base_voice": "amy",
    # Piper fine-tune learning rate. The lessac base checkpoint is at ~1.5e-4
    # (2e-4 decayed over 2000+ epochs); this restarts lower for gentler,
    # smoother late-stage convergence.
    "piper_learning_rate": 5e-5,
    # How many MORE epochs a training run fine-tunes for (on top of the checkpoint's own).
    "piper_epochs": 1000,
    # Per aspect: "base" = keep the base voice's, "reference" = learn from the
    # clips. See modules/piper.MATCH_ASPECTS.
    "piper_match_timbre": "reference",
    "piper_match_rhythm": "base",
    "piper_match_pronunciation": "base",
    # Include inflated clips in the training set.
    "piper_use_inflated": True,
}


def load_settings() -> dict:
    if _SETTINGS_FILE.exists():
        try:
            stored = json.loads(_SETTINGS_FILE.read_text())
            if "piper_use_padding" in stored:         # renamed when "padding" became "inflated"
                stored.setdefault("piper_use_inflated", stored.pop("piper_use_padding"))
            return {**_DEFAULTS, **stored}
        except Exception:
            pass
    return dict(_DEFAULTS)


def save_settings(data: dict) -> None:
    _SETTINGS_FILE.write_text(json.dumps(data, indent=2))


def get_default_base_voice() -> str:
    return load_settings()["default_base_voice"]


def set_default_base_voice(key: str) -> None:
    data = load_settings()
    data["default_base_voice"] = key
    save_settings(data)


def get_piper_learning_rate() -> float:
    return float(load_settings()["piper_learning_rate"])


def set_piper_learning_rate(lr: float) -> None:
    data = load_settings()
    data["piper_learning_rate"] = float(lr)
    save_settings(data)


def get_piper_epochs() -> int:
    return max(1, int(load_settings()["piper_epochs"]))


def set_piper_epochs(n: int) -> None:
    data = load_settings()
    data["piper_epochs"] = max(1, int(n))
    save_settings(data)


def get_piper_match() -> dict:
    d = load_settings()
    return {a: d[f"piper_match_{a}"] for a in ("timbre", "rhythm", "pronunciation")}


def set_piper_match(aspect: str, choice: str) -> None:
    data = load_settings()
    data[f"piper_match_{aspect}"] = choice
    save_settings(data)


def get_piper_use_inflated() -> bool:
    return bool(load_settings()["piper_use_inflated"])


def set_piper_use_inflated(use: bool) -> None:
    data = load_settings()
    data["piper_use_inflated"] = bool(use)
    save_settings(data)
