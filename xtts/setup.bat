@echo off
setlocal enabledelayedexpansion

echo ============================================================
echo XTTS-v2 Voice Clone Pipeline — Setup (6GB VRAM Optimization)
echo ============================================================
echo.

:: --- Check if venv exists ---
if not exist "..\\.venv" (
    echo [ERROR] Virtual environment not found in root!
    echo Run the main setup.bat first.
    exit /b 1
)

set PYTHON=..\\.venv\Scripts\python.exe
set PIP=..\\.venv\Scripts\pip.exe

:: --- Install XTTS Dependencies ---
echo [1/3] Installing XTTS-v2 core dependencies...
echo (This may take a few minutes...)
"%PIP%" install TTS deepspeed==0.16.2 accelerate pydub --no-cache-dir

:: --- Check for deepspeed version conflict (fix for Windows) ---
:: On Windows, deepspeed often needs manual building or specific whls if not careful.
:: We will use regular coqui-tts first, and only use deepspeed if really needed.

:: --- Download XTTS-v2 Base Model ---
echo [2/3] Downloading XTTS-v2 Base Model (~2.5 GB)...
"%PYTHON%" -c "from TTS.utils.manage_config import Config; from TTS.utils.downloaders import download_xtts_models; download_xtts_models()"

:: --- Create Folders ---
echo [3/3] Creating xtts_output/ folder...
if not exist "xtts_output" mkdir "xtts_output"

echo.
echo ============================================================
echo XTTS-v2 Setup Complete!
echo.
echo Usage: 
echo   - Train:   python xtts/train.py
echo   - Generate: python xtts/generate.py --text "Hello world"
echo ============================================================
pause
