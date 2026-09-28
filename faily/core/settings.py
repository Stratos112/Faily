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
    "default_base_voice": "lessac",
}


def load_settings() -> dict:
    if _SETTINGS_FILE.exists():
        try:
            return {**_DEFAULTS, **json.loads(_SETTINGS_FILE.read_text())}
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
