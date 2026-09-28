"""Hugging Face access token — pasted once via the startup dialog or Settings,
stored locally, and reused for any gated-repo download. Falls back to the
HF_TOKEN env var (e.g. set in docker-compose) when nothing is saved."""
import os

from faily.core.settings import config_file

_TOKEN_FILE = config_file("hf_token.txt")


def has_hf_token() -> bool:
    return bool(get_hf_token())


def get_hf_token() -> str | None:
    if _TOKEN_FILE.exists():
        tok = _TOKEN_FILE.read_text().strip()
        if tok:
            return tok
    return os.environ.get("HF_TOKEN", "").strip() or None


def save_hf_token(token: str) -> None:
    _TOKEN_FILE.write_text(token.strip())


def clear_hf_token() -> None:
    _TOKEN_FILE.unlink(missing_ok=True)
