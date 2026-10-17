import os
from pathlib import Path

# HuggingFace cache under models/ — persisted both in the dev workspace and in
# the container (models/ is a volume), unlike the default ~/.cache/huggingface.
# Must be set before anything imports huggingface_hub, hence here, first thing.
os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parent.parent / "models" / "hf"))

from faily.core.model_manager import manager

__all__ = ["manager"]
