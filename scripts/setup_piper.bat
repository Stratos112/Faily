@echo off
setlocal enabledelayedexpansion

set SCRIPT_DIR=%~dp0
set PROJECT_DIR=%SCRIPT_DIR%..
set VENV=%PROJECT_DIR%\piper_venv
set CKPT_DIR=%PROJECT_DIR%\piper_checkpoints

:: ── Python version (piper-train needs 3.9-3.11) ────────────────────────────
set PYVER=
for %%v in (3.11 3.10 3.9) do (
    if not defined PYVER (
        py -%%v --version >nul 2>&1
        if not errorlevel 1 set PYVER=%%v
    )
)
if not defined PYVER (
    echo ERROR: Python 3.9-3.11 required for piper-train.
    echo Install from https://python.org then re-run this script.
    exit /b 1
)
echo Using Python %PYVER%

:: ── Create venv ─────────────────────────────────────────────────────────────
if exist "%VENV%\Scripts\python.exe" (
    echo Piper venv already exists — skipping creation.
    goto :download
)
echo Creating venv at %VENV%
py -%PYVER% -m venv "%VENV%"
call "%VENV%\Scripts\activate.bat"
python -m pip install --upgrade pip
pip install piper-train piper-tts
echo Venv ready.

:download
:: ── Download base checkpoint (en_US-lessac-medium) ─────────────────────────
if not exist "%CKPT_DIR%" mkdir "%CKPT_DIR%"

set CKPT=%CKPT_DIR%\epoch=2164-step=1355540.ckpt
if not exist "%CKPT%" (
    echo Downloading base checkpoint (~400 MB)...
    curl -L --progress-bar -o "%CKPT%" ^
      "https://huggingface.co/datasets/rhasspy/piper-checkpoints/resolve/main/en/en_US/lessac/medium/epoch=2164-step=1355540.ckpt"
) else (
    echo Base checkpoint already present.
)

set CFG=%CKPT_DIR%\en_US-lessac-medium.onnx.json
if not exist "%CFG%" (
    echo Downloading voice config...
    curl -L -o "%CFG%" ^
      "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json"
) else (
    echo Voice config already present.
)

echo.
echo Piper setup complete.
echo   Venv:       %VENV%
echo   Checkpoint: %CKPT%
echo   Config:     %CFG%
