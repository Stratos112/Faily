@echo off
setlocal enabledelayedexpansion

set SCRIPT_DIR=%~dp0
set PROJECT_DIR=%SCRIPT_DIR%..
set VENV=%PROJECT_DIR%\.venv

echo ============================================================
echo  Faily — main environment setup
echo ============================================================
echo.

:: ── Python check ──────────────────────────────────────────────────────────────
py --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found via the Windows launcher ^(py^).
    echo Install Python 3.12+ from https://www.python.org/
    echo   ^> Check "Add python.exe to PATH" during install.
    pause & exit /b 1
)
for /f "tokens=2" %%v in ('py --version 2^>^&1') do set PYVER=%%v
echo Python: %PYVER%

:: ── Git check ^(required for GitHub package installs^) ─────────────────────────
git --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Git not found. Install from https://git-scm.com/
    pause & exit /b 1
)
echo Git: OK

echo.

:: ── Create venv ───────────────────────────────────────────────────────────────
if exist "%VENV%\Scripts\python.exe" (
    echo Virtual env already exists — skipping creation.
) else (
    echo Creating virtual environment at .venv ...
    py -m venv "%VENV%"
    if errorlevel 1 ( echo ERROR: venv creation failed. & pause & exit /b 1 )
)

:: ── Upgrade pip ───────────────────────────────────────────────────────────────
echo Upgrading pip...
"%VENV%\Scripts\pip" install --upgrade pip --quiet

:: ── PyTorch cu128 for Blackwell GPUs (RTX 50xx series) ───────────────────────
:: Skip if already installed with cu128 to avoid a slow re-download.
"%VENV%\Scripts\python" -c "import torch; assert 'cu128' in torch.__version__" >nul 2>&1
if errorlevel 1 (
    echo Installing PyTorch 2.x cu128 ^(~2.5 GB — this will take a while^)...
    "%VENV%\Scripts\pip" install torch torchaudio ^
        --index-url https://download.pytorch.org/whl/cu128
    if errorlevel 1 ( echo ERROR: PyTorch install failed. & pause & exit /b 1 )
) else (
    echo PyTorch cu128 already installed.
)

:: ── Core Faily package ────────────────────────────────────────────────────────
:: Installs: nicegui, transformers, diffusers, accelerate, soundfile,
::           numpy, scipy, speechbrain, huggingface_hub
echo Installing Faily core dependencies...
"%VENV%\Scripts\pip" install -e "%PROJECT_DIR%" --quiet
if errorlevel 1 ( echo ERROR: Faily core install failed. & pause & exit /b 1 )

:: ── Voice cloning backends (CLONE / ONE SHOT / SPEAK stage-2) ─────────────────
echo Installing voice cloning backends...
echo   coqui-tts  ^(XTTS v2 + FreeVC^)...
"%VENV%\Scripts\pip" install coqui-tts --quiet
if errorlevel 1 ( echo WARNING: coqui-tts install failed — XTTS and FreeVC will not work. )

echo   f5-tts...
"%VENV%\Scripts\pip" install f5-tts --quiet
if errorlevel 1 ( echo WARNING: f5-tts install failed. )

echo   chatterbox-tts...
"%VENV%\Scripts\pip" install chatterbox-tts --quiet
if errorlevel 1 ( echo WARNING: chatterbox-tts install failed. )

:: ── Expression engines (SPEAK stage-1) ───────────────────────────────────────
echo Installing expression engines...
echo   parler-tts...
"%VENV%\Scripts\pip" install parler-tts --quiet
if errorlevel 1 ( echo WARNING: parler-tts install failed. )

:: ── OpenVoice v2 ^(SPEAK stage-2 voice conversion^) ──────────────────────────
:: --no-deps avoids pinned version conflicts (gradio 3.x, numpy 1.22, librosa 0.9, etc.)
:: that would downgrade packages already installed at newer compatible versions.
echo   OpenVoice v2 ^(--no-deps to skip pinned version conflicts^)...
"%VENV%\Scripts\pip" install --no-deps "git+https://github.com/myshell-ai/OpenVoice.git"
if errorlevel 1 ( echo WARNING: OpenVoice install failed. ) else (
    :: Runtime deps that are genuinely absent ^(skipping pinned conflicts — newer versions work^)
    "%VENV%\Scripts\pip" install wavmark resampy faster-whisper cn2an eng_to_ipa langid jieba --quiet
)

:: ── Seed-VC ^(SPEAK stage-2 zero-shot voice conversion^) ───────────────────────
:: Clone repo — used as a local import, not installed as a package.
echo   Seed-VC repo...
set SEEDVC_DIR=%PROJECT_DIR%\models\vc\seed-vc
if not exist "%SEEDVC_DIR%" (
    git clone https://github.com/Plachtaa/seed-vc.git "%SEEDVC_DIR%"
    if errorlevel 1 ( echo WARNING: Seed-VC clone failed. )
) else (
    echo   Seed-VC repo already present.
)

:: Seed-VC deps — install selectively; the repo's requirements.txt pins torch==2.4,
:: numpy==1.26.4, transformers==4.46.3 which would downgrade our cu128 stack.
:: Only install packages that aren't already covered by the core Faily install.
echo   Seed-VC deps ^(bigvgan, munch, einops, descript-audio-codec, resemblyzer, pydub, hydra-core, sounddevice^)...
"%VENV%\Scripts\pip" install ^
    bigvgan ^
    munch ^
    einops ^
    descript-audio-codec ^
    resemblyzer ^
    pydub ^
    hydra-core ^
    python-dotenv ^
    sounddevice ^
    jiwer ^
    --quiet
if errorlevel 1 ( echo WARNING: one or more Seed-VC deps failed — check output above. )

:: ── Patch bigvgan + huggingface_hub ────────────────────────────────────────────
:: bigvgan._from_pretrained declares proxies/resume_download as required kwargs,
:: but HF Hub >=0.24 no longer passes them. These one-time patches fix the mismatch.
echo Patching bigvgan...
"%VENV%\Scripts\python" "%SCRIPT_DIR%patch_bigvgan.py"
if errorlevel 1 ( echo WARNING: bigvgan patch failed — Seed-VC may not load. )

echo Patching huggingface_hub hub_mixin...
"%VENV%\Scripts\python" "%SCRIPT_DIR%patch_hub_mixin.py"
if errorlevel 1 ( echo WARNING: hub_mixin patch failed — Seed-VC may not load. )

:: ── Windows environment variable ──────────────────────────────────────────────
:: Suppresses HuggingFace symlink warnings (irrelevant on Windows)
setx HF_HUB_DISABLE_SYMLINKS_WARNING 1 >nul 2>&1

echo.
echo ============================================================
echo  Setup complete.
echo ============================================================
echo.
echo  Verify:
echo    %VENV%\Scripts\python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.__version__)"
echo.
echo  Run Faily:
echo    %VENV%\Scripts\python "%PROJECT_DIR%\main.py"
echo.
echo  For Piper TTS training (CHARACTER tab), also run:
echo    scripts\setup_piper.bat
echo.
pause
