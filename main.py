import os
import sys
from pathlib import Path


def _project_python(root: Path | None = None) -> Path | None:
    """The project's own interpreter (.venv), if it has one."""
    root = root or Path(__file__).resolve().parent
    for rel in (".venv/bin/python", ".venv/Scripts/python.exe"):
        if (root / rel).exists():
            return root / rel
    return None


def _use_project_python() -> None:
    """`python3 main.py` from a plain shell picks the system Python, which has none of the
    dependencies. When we are not inside any virtualenv and the project has a .venv, run
    under that instead. FAILY_NO_REEXEC=1 turns this off."""
    if os.environ.get("FAILY_NO_REEXEC") or sys.prefix != sys.base_prefix:
        return
    py = _project_python()
    if py is not None:
        os.execv(str(py), [str(py), *sys.argv])


if __name__ == "__main__":
    _use_project_python()

from faily.ui.app import run

if __name__ in {"__main__", "__mp_main__"}:
    run()
