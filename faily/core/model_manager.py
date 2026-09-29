from pathlib import Path
from typing import Any, Callable
import torch


def _preload_npp():
    # torchcodec's cu128 build links NVIDIA NPP (libnpp*.so.12) but has no
    # rpath to the nvidia-npp-cu12 wheel's lib dir, so it fails to load even
    # when installed. torch preloads its own nvidia/* libs this same way;
    # once NPP is loaded RTLD_GLOBAL, the dynamic linker resolves torchcodec's
    # dependency against it by soname.
    import ctypes
    try:
        import nvidia.npp
        lib_dir = Path(list(nvidia.npp.__path__)[0]) / "lib"
    except Exception:
        return
    pending = sorted(lib_dir.glob("libnpp*.so.*"))
    for _ in range(len(pending)):  # retry until inter-library deps are satisfied
        still = []
        for lib in pending:
            try:
                ctypes.CDLL(str(lib), mode=ctypes.RTLD_GLOBAL)
            except OSError:
                still.append(lib)
        if not still or len(still) == len(pending):
            break
        pending = still

_preload_npp()

TTS_MODELS_DIR = Path("models/tts")
SFX_MODELS_DIR = Path("models/sfx")
VC_MODELS_DIR  = Path("models/vc")
TTS_MODELS_DIR.mkdir(parents=True, exist_ok=True)
SFX_MODELS_DIR.mkdir(parents=True, exist_ok=True)
VC_MODELS_DIR.mkdir(parents=True, exist_ok=True)


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
