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
echo   kokoro...
"%VENV%\Scripts\pip" install kokoro --quiet
if errorlevel 1 ( echo WARNING: kokoro install failed. )

echo   parler-tts...
"%VENV%\Scripts\pip" install parler-tts --quiet
if errorlevel 1 ( echo WARNING: parler-tts install failed. )

:: MeloTTS has two problematic deps on Windows + Python 3.14:
::   mecab-python3 / fugashi — need MeCab C headers (only used for JP/ZH tokenization)
::   tokenizers — MeloTTS pins an old version with no cp314 wheel (needs Rust to compile)
:: Fix: pre-install current tokenizers ^(has cp314 wheel^), then install MeloTTS --no-deps,
:: then add only the English-compatible runtime deps manually.
echo   tokenizers ^(pre-install latest to avoid Rust compiler^)...
"%VENV%\Scripts\pip" install tokenizers --quiet

echo   MeloTTS ^(--no-deps to skip mecab/fugashi; English-only^)...
"%VENV%\Scripts\pip" install --no-deps "git+https://github.com/myshell-ai/MeloTTS.git"
if errorlevel 1 ( echo WARNING: MeloTTS install failed. ) else (
    echo   MeloTTS English deps ^(librosa, inflect, langdetect^)...
    "%VENV%\Scripts\pip" install librosa inflect langdetect --quiet
)

:: MeloTTS NLTK data (needed at runtime for English text processing)
echo   MeloTTS NLTK data...
"%VENV%\Scripts\python" -c ^
    "import nltk; nltk.download('averaged_perceptron_tagger_eng', quiet=True)" ^
    >nul 2>&1
if errorlevel 1 ( echo WARNING: NLTK download failed — run manually: python -c "import nltk; nltk.download('averaged_perceptron_tagger_eng')" )

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
echo   Seed-VC...
set SEEDVC_DIR=%PROJECT_DIR%\models\vc\seed-vc
if not exist "%SEEDVC_DIR%" (
    git clone https://github.com/Plachtaa/seed-vc.git "%SEEDVC_DIR%"
)
if exist "%SEEDVC_DIR%\requirements.txt" (
    "%VENV%\Scripts\pip" install -r "%SEEDVC_DIR%\requirements.txt" --quiet
    if errorlevel 1 ( echo WARNING: Seed-VC requirements install failed. )
)

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
