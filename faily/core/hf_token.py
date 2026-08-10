"""Hugging Face access token — pasted once via the startup dialog or Settings,
stored locally, and reused for any gated-repo download."""
from pathlib import Path

_TOKEN_FILE = Path(__file__).parent.parent.parent / "hf_token.txt"


def has_hf_token() -> bool:
    return bool(get_hf_token())


def get_hf_token() -> str | None:
    if _TOKEN_FILE.exists():
        tok = _TOKEN_FILE.read_text().strip()
        return tok or None
    return None


def save_hf_token(token: str) -> None:
    _TOKEN_FILE.write_text(token.strip())


def clear_hf_token() -> None:
    _TOKEN_FILE.unlink(missing_ok=True)
