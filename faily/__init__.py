import os
from pathlib import Path

# HuggingFace cache under models/ — persisted both in the dev workspace and in
# the container (models/ is a volume), unlike the default ~/.cache/huggingface.
# Must be set before anything imports huggingface_hub, hence here, first thing.
os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parent.parent / "models" / "hf"))

# Hand the token saved in Settings (config/hf_token.txt) to huggingface_hub, which only
# looks at HF_TOKEN / its own login file. Without this every Hub request — including
# the "is my cache up to date?" check on each model load — goes out unauthenticated
# and prints "You are sending unauthenticated requests to the HF Hub".
try:
    from faily.core.hf_token import get_hf_token as _get_hf_token
    _tok = _get_hf_token()
    if _tok:
        os.environ.setdefault("HF_TOKEN", _tok)
except Exception:                       # a bad config file must never stop the app starting
    pass

from faily.core.model_manager import manager

__all__ = ["manager"]
