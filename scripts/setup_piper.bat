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

:: ── espeak-ng check ──────────────────────────────────────────────────────────
if not exist "%ESPEAK_DIR%\include\espeak-ng\speak_lib.h" (
    echo ERROR: espeak-ng headers not found at "%ESPEAK_DIR%"
    echo If you installed to a different path, edit ESPEAK_DIR at the top of this script.
    pause
    exit /b 1
)
echo Found espeak-ng at %ESPEAK_DIR%

:: ── Create venv ──────────────────────────────────────────────────────────────
if exist "%VENV%\Scripts\python.exe" (
    echo Piper venv already exists.
    goto :install_packages
)
echo Creating venv...
py -3.11 -m venv "%VENV%"

:install_packages
:: Downgrade pip so pytorch-lightning 1.7.x metadata is accepted
echo Downgrading pip to allow pytorch-lightning 1.7.x...
"%VENV%\Scripts\pip" install "pip<24.1" --quiet

:: ── piper-phonemize (compile from source using local espeak-ng) ──────────────
:: Point cmake at the espeak-ng install so it finds headers and libs
echo Installing piper-phonemize (compiling from source)...
set CMAKE_PREFIX_PATH=%ESPEAK_DIR%
set ESPEAK_NG_DIR=%ESPEAK_DIR%
"%VENV%\Scripts\pip" install piper-phonemize
if errorlevel 1 (
    echo.
    echo ERROR: piper-phonemize build failed.
    echo Make sure Visual Studio Build Tools ^(C++ workload^) are installed and
    echo that espeak-ng is at "%ESPEAK_DIR%".
    pause
    exit /b 1
)

:: ── piper-train from GitHub source ───────────────────────────────────────────
echo Installing piper-train...
"%VENV%\Scripts\pip" install "piper-train @ git+https://github.com/rhasspy/piper.git#subdirectory=src/python"
if errorlevel 1 (
    echo ERROR: piper-train install failed.
    pause
    exit /b 1
)

:: ── piper-tts (inference binary) ─────────────────────────────────────────────
echo Installing piper-tts...
"%VENV%\Scripts\pip" install piper-tts --quiet

:: ── Replace torch with cu128 for RTX 5070 Ti (Blackwell) ────────────────────
:: piper-train pulls torch 1.13.1+cu117 which cannot address Blackwell GPUs.
echo Replacing torch with cu128 build...
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
echo   %VENV%\Scripts\python -c "import piper_train; import torch; print('OK  torch', torch.__version__, ' CUDA:', torch.cuda.is_available())"
