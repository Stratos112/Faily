@echo off
setlocal enabledelayedexpansion

set SCRIPT_DIR=%~dp0
set PROJECT_DIR=%SCRIPT_DIR%..
set VENV=%PROJECT_DIR%\piper_venv
set CKPT_DIR=%PROJECT_DIR%\piper_checkpoints

:: Default espeak-ng install path — change if you installed elsewhere
set ESPEAK_DIR=C:\Program Files\eSpeak NG

:: ── Python 3.11 check ────────────────────────────────────────────────────────
py -3.11 --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python 3.11 not found.
    echo Install from: https://www.python.org/downloads/release/python-3119/
    echo   ^> Check "Add python.exe to PATH" during install.
    pause
    exit /b 1
)
echo Using Python 3.11

:: ── espeak-ng check (binary only — no headers needed) ────────────────────────
if not exist "%ESPEAK_DIR%\espeak-ng.exe" (
    echo ERROR: espeak-ng not found at "%ESPEAK_DIR%"
    echo Download installer from: https://github.com/espeak-ng/espeak-ng/releases
    echo If you installed elsewhere, edit ESPEAK_DIR at the top of this script.
    pause
    exit /b 1
)
echo Found espeak-ng at %ESPEAK_DIR%

:: Add espeak-ng to PATH so phonemizer can find it
set "PATH=%ESPEAK_DIR%;%PATH%"

:: ── Create venv ──────────────────────────────────────────────────────────────
if exist "%VENV%\Scripts\python.exe" (
    echo Piper venv already exists — checking for updates.
    goto :install_packages
)
echo Creating piper venv...
py -3.11 -m venv "%VENV%"

:install_packages
:: Downgrade pip so pytorch-lightning 1.7.x metadata is accepted
echo Downgrading pip to allow pytorch-lightning 1.7.x...
"%VENV%\Scripts\pip" install "pip<24.1" --quiet

:: ── piper-tts (inference binary) ──────────────────────────────────────────────
echo Installing piper-tts...
"%VENV%\Scripts\pip" install piper-tts --quiet

:: ── phonemizer (replaces piper-phonemize — no C compilation needed) ──────────
:: phonemizer calls the espeak-ng binary we already have; the piper_phonemize
:: shim is written into site-packages automatically on first training run.
echo Installing phonemizer...
"%VENV%\Scripts\pip" install phonemizer --quiet
if errorlevel 1 (
    echo ERROR: phonemizer install failed.
    pause
    exit /b 1
)

:: ── piper-train runtime deps (manual, so we can skip piper-phonemize) ─────────
echo Installing piper-train dependencies...
"%VENV%\Scripts\pip" install cython --quiet
"%VENV%\Scripts\pip" install librosa --quiet
"%VENV%\Scripts\pip" install "pytorch-lightning~=1.7.0" --quiet
if errorlevel 1 (
    echo ERROR: pytorch-lightning install failed.
    pause
    exit /b 1
)

:: ── piper-train from GitHub (--no-deps skips piper-phonemize requirement) ─────
echo Installing piper-train ^(no-deps^)...
"%VENV%\Scripts\pip" install --no-deps ^
    "piper-train @ git+https://github.com/rhasspy/piper.git#subdirectory=src/python"
if errorlevel 1 (
    echo ERROR: piper-train install failed.
    pause
    exit /b 1
)

:: ── Replace torch with cu128 for RTX 5070 Ti (Blackwell) ────────────────────
:: piper-train pulls torch 1.13.1+cu117 which cannot run on Blackwell GPUs.
echo Replacing torch with cu128 build for RTX 5070 Ti...
"%VENV%\Scripts\pip" install torch --index-url https://download.pytorch.org/whl/cu128 --force-reinstall --quiet
if errorlevel 1 (
    echo WARNING: torch cu128 install failed — training will fall back to CPU.
)

:: ── Base checkpoint ───────────────────────────────────────────────────────────
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
echo Setup complete.
echo.
echo Verify with:
echo   %VENV%\Scripts\python -c "import phonemizer, piper_train; import torch; print('OK  torch', torch.__version__, ' CUDA:', torch.cuda.is_available())"
