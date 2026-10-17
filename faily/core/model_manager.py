import gc
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable
import torch

log = logging.getLogger(__name__)


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
    """Lazy model cache with least-recently-used eviction under VRAM pressure.

    Every backend is 1–6 GB of VRAM; keeping all of them resident for the
    life of the server runs a 16 GB card out of memory after a handful of
    backends have been tried. Before loading a new model, evict the least
    recently used ones until at least FAILY_VRAM_RESERVE_GB is free (default
    6 — just above the biggest single pipeline measured, Parler + Seed-VC).
    """

    def __init__(self):
        self._cache: dict[str, Any] = {}
        self._last_used: dict[str, float] = {}
        # Generations run in worker threads (nicegui run.io_bound), so two
        # clients can hit load() at once; reentrant because a loader may
        # itself load a sub-model.
        self._lock = threading.RLock()
        self.vram_reserve_gb = float(os.environ.get("FAILY_VRAM_RESERVE_GB", "6"))
        self.device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )

    def load(self, key: str, loader: Callable[[], Any]) -> Any:
        with self._lock:
            if key not in self._cache:
                self._make_room()
                try:
                    self._cache[key] = loader()
                except torch.cuda.OutOfMemoryError:
                    # The reserve guessed low (or another process grabbed
                    # VRAM) — drop everything else and retry once.
                    log.warning("OOM loading %s — unloading %s and retrying", key, self.loaded)
                    self.unload_all()
                    self._cache[key] = loader()
            self._last_used[key] = time.monotonic()
            return self._cache[key]

    def _free_vram_gb(self) -> float | None:
        if self.device != "cuda":
            return None
        free, _total = torch.cuda.mem_get_info()
        return free / 2**30

    def _make_room(self):
        free = self._free_vram_gb()
        while free is not None and free < self.vram_reserve_gb and self._cache:
            victim = min(self._cache, key=lambda k: self._last_used.get(k, 0.0))
            log.info("VRAM %.1f GB free < %.1f GB reserve — unloading %s", free, self.vram_reserve_gb, victim)
            self.unload(victim)
            free = self._free_vram_gb()

    def unload(self, key: str):
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                self._last_used.pop(key, None)
                self._release()

    def unload_all(self):
        with self._lock:
            self._cache.clear()
            self._last_used.clear()
            self._release()

    @staticmethod
    def _release():
        # Models often sit in reference cycles; collect before asking the CUDA
        # allocator to hand the memory back, or empty_cache() frees nothing.
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @property
    def loaded(self) -> list[str]:
        return list(self._cache.keys())


manager = ModelManager()
