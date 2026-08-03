@echo off
setlocal enabledelayedexpansion

set SCRIPT_DIR=%~dp0
set PROJECT_DIR=%SCRIPT_DIR%..
set VENV=%PROJECT_DIR%\piper_venv
set CKPT_DIR=%PROJECT_DIR%\piper_checkpoints

:: ── Python 3.11 check ────────────────────────────────────────────────────────
:: piper-train requires Python 3.9-3.11. If py -3.11 is not found, download it
:: from https://www.python.org/downloads/release/python-3119/ (Windows installer)
:: then re-run this script.
py -3.11 --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python 3.11 not found.
    echo Install from: https://www.python.org/downloads/release/python-3119/
    echo   ^> Check "Add python.exe to PATH" during install.
    echo Then re-run this script.
    pause
    exit /b 1
)
echo Using Python 3.11

:: ── Create venv ─────────────────────────────────────────────────────────────
if exist "%VENV%\Scripts\python.exe" (
    echo Piper venv already exists — skipping creation.
    goto :install_train
)
echo Creating venv...
py -3.11 -m venv "%VENV%"

:install_train
:: Downgrade pip to <24.1 so pytorch-lightning 1.7.x metadata is accepted
echo Downgrading pip to allow pytorch-lightning 1.7.x install...
"%VENV%\Scripts\pip" install "pip<24.1" --quiet

:: Install piper-tts (inference binary) and piper-train from GitHub source
echo Installing piper-tts and piper-train (this may take a few minutes)...
"%VENV%\Scripts\pip" install piper-tts
"%VENV%\Scripts\pip" install "piper-train @ git+https://github.com/rhasspy/piper.git#subdirectory=src/python"

:: ── Replace torch with cu128 for RTX 5070 Ti (Blackwell sm_120) ─────────────
:: piper-train pulls torch 1.13.1+cu117 which cannot address Blackwell GPUs.
:: Force-reinstall torch 2.x with cu128 (pytorch-lightning 1.7 is compatible).
echo Replacing torch with cu128 build for Blackwell GPU support...
"%VENV%\Scripts\pip" install torch --index-url https://download.pytorch.org/whl/cu128 --force-reinstall --quiet

:: ── Download base checkpoint ─────────────────────────────────────────────────
if not exist "%CKPT_DIR%" mkdir "%CKPT_DIR%"

set CKPT=%CKPT_DIR%\epoch=2164-step=1355540.ckpt
if not exist "%CKPT%" (
    echo Downloading base checkpoint ~400 MB...
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
echo.
echo Verify with:
echo   %VENV%\Scripts\python -c "import piper_train; import torch; print('OK torch', torch.__version__, 'CUDA:', torch.cuda.is_available())"
