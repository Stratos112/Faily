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
:: Sanity check before reuse: a venv created by a different interpreter left
:: a Scripts\python.exe on disk that exists but fails to launch. Detect and
:: rebuild instead of silently trying to use a broken venv.
if exist "%VENV%\Scripts\python.exe" (
    "%VENV%\Scripts\python.exe" --version >nul 2>&1
    if errorlevel 1 (
        echo Existing piper venv is broken - rebuilding it...
        rmdir /s /q "%VENV%"
        goto :create_venv
    )
    echo Piper venv already exists — checking for updates.
    goto :install_packages
)

:create_venv
echo Creating piper venv...
py -3.11 -m venv "%VENV%"

:install_packages
:: Downgrade pip so pytorch-lightning 1.7.x metadata is accepted.
:: pip 24.1+ blocks self-modification through the pip script — must use python -m pip.
echo Downgrading pip to allow pytorch-lightning 1.7.x...
"%VENV%\Scripts\python" -m pip install "pip<24.1" --quiet
if errorlevel 1 (
    echo WARNING: pip downgrade failed — see note above.
    echo   Run manually if errors follow:
    echo   "%VENV%\Scripts\python" -m pip install "pip^<24.1"
)

:: Remove any existing pytorch-lightning before reinstalling.
:: pip 24.1+ refuses to process packages already in site-packages with
:: invalid requirement specs (>=1.9.*), so a stale install blocks everything.
"%VENV%\Scripts\python" -m pip uninstall pytorch-lightning -y 2>nul

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
"%VENV%\Scripts\pip" install six --quiet
:: pytorch-lightning 1.7.x declares torchmetrics>=0.7.0 with no upper bound, so
:: pip grabs the newest release, which dropped the private _compare_version
:: helper 1.7.x imports at startup. Pin a version still compatible with 1.7.x.
"%VENV%\Scripts\pip" install "torchmetrics==0.11.4" --quiet
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

:: piper-train's setup.py doesn't declare norm_audio/models/*.onnx as package
:: data, so pip's wheel build silently drops it even though it's a real file
:: in the repo. Without it, preprocessing crashes trying to load the VAD model.
set VAD_DIR=%VENV%\Lib\site-packages\piper_train\norm_audio\models
set VAD_FILE=%VAD_DIR%\silero_vad.onnx
if not exist "%VAD_FILE%" (
    echo Downloading silero_vad.onnx ^(missing from piper-train package^)...
    if not exist "%VAD_DIR%" mkdir "%VAD_DIR%"
    curl -L -o "%VAD_FILE%" ^
      "https://raw.githubusercontent.com/rhasspy/piper/master/src/python/piper_train/norm_audio/models/silero_vad.onnx"
)

:: After this step pip's resolver will report several dependency conflicts.
:: These are ALL expected and handled — do not treat them as errors:
::
::   piper-phonemize~=1.1.0 not installed
::       ^^ Faily writes a pure-Python shim into the venv on first training run.
::
::   cython<1 conflict ^(you have cython 3.x^)
::       ^^ Cython is a build tool only. piper-train never calls it at runtime.
::
::   torch<2 conflict ^(you have torch 2.x+cu128^)
::       ^^ piper-train's pin is conservative; torch 2.x is backward-compatible.
::
::   pytorch-lightning DEPRECATION warning
::       ^^ Non-standard ">=1.9.*" specifier. Treated as warning ^(not error^)
::          because we downgraded pip to ^<24.1 above.
::
:: The verification command at the end is the real pass/fail signal.
echo.
echo ^(Expected pip resolver warnings above — see comments in script for details^)
echo.

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
