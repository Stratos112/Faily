import json
from pathlib import Path

_SETTINGS_FILE = Path(__file__).parent.parent.parent / "faily_settings.json"
_DEFAULTS: dict = {
    "download_dir": str(Path.home() / "Downloads"),
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


def get_download_dir() -> Path:
    return Path(load_settings()["download_dir"])


def get_default_base_voice() -> str:
    return load_settings()["default_base_voice"]


def set_default_base_voice(key: str) -> None:
    data = load_settings()
    data["default_base_voice"] = key
    save_settings(data)
