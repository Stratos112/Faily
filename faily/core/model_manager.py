from pathlib import Path
from typing import Any, Callable
import torch

TTS_MODELS_DIR = Path("models/tts")
SFX_MODELS_DIR = Path("models/sfx")
TTS_MODELS_DIR.mkdir(parents=True, exist_ok=True)
SFX_MODELS_DIR.mkdir(parents=True, exist_ok=True)


class ModelManager:
    def __init__(self):
        self._cache: dict[str, Any] = {}
        self.device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )

    def load(self, key: str, loader: Callable[[], Any]) -> Any:
        if key not in self._cache:
            self._cache[key] = loader()
        return self._cache[key]

    def unload(self, key: str):
        if key in self._cache:
            del self._cache[key]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def unload_all(self):
        self._cache.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @property
    def loaded(self) -> list[str]:
        return list(self._cache.keys())


manager = ModelManager()
