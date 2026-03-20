@echo off
REM ============================================================
REM Voice Clone Pipeline - Windows Setup Script
REM ============================================================
REM Model: Chatterbox Turbo Fine-Tuning
REM Uses: uv (package manager) + Python 3.11
REM Run: .\setup.bat
REM ============================================================

echo ============================================================
echo    Voice Clone Pipeline - Chatterbox Turbo Setup
echo ============================================================
echo.

REM --- Check uv ---
uv --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] uv not found. Install: https://docs.astral.sh/uv/
    echo         Or: pip install uv
    pause
    exit /b 1
)
echo [OK] uv found.

REM --- Check FFmpeg ---
ffmpeg -version >nul 2>&1
if %errorlevel% neq 0 (
    echo [WARNING] FFmpeg not found!
    echo           If you just installed it, RESTART your terminal first.
    echo           Install: winget install Gyan.FFmpeg
    echo.
)

REM --- Check Git ---
git --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Git not found. Install: https://git-scm.com
    pause
    exit /b 1
)
echo [OK] Git found.
echo.

REM --- Create venv with Python 3.11 if not exists ---
if not exist ".venv" (
    echo [*] Creating virtual environment with Python 3.11...
    uv venv -p 3.11
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create venv.
        echo         Make sure Python 3.11 is installed: uv python install 3.11
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created.
) else (
    echo [OK] Virtual environment already exists.
)
echo.

REM --- Install PyTorch with CUDA 12.1 ---
echo [*] Installing PyTorch with CUDA 12.1 support...
echo     (This may take several minutes on first run)
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
if %errorlevel% neq 0 (
    echo [WARNING] CUDA PyTorch failed. Trying default...
    uv pip install torch torchaudio
)
echo.

REM --- Install project dependencies ---
echo [*] Installing project dependencies...
uv pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [ERROR] Dependency installation failed. See errors above.
    pause
    exit /b 1
)
echo.

REM --- Clone Chatterbox Fine-Tuning Toolkit ---
if not exist "chatterbox-finetuning" (
    echo [*] Cloning Chatterbox fine-tuning toolkit...
    git clone https://github.com/gokhaneraslan/chatterbox-finetuning.git
    if %errorlevel% neq 0 (
        echo [ERROR] Git clone failed. Check your internet connection.
        pause
        exit /b 1
    )
    echo [OK] Toolkit cloned.
    echo.
    echo [*] Installing toolkit dependencies...
    uv pip install -r chatterbox-finetuning\requirements.txt
) else (
    echo [OK] Chatterbox fine-tuning toolkit already present.
)
echo.

REM --- Create project directories ---
echo [*] Creating project directories...
if not exist "data\raw" mkdir data\raw
if not exist "data\processed" mkdir data\processed
if not exist "MyTTSDataset\wavs" mkdir MyTTSDataset\wavs
if not exist "pretrained_models" mkdir pretrained_models
if not exist "chatterbox_output" mkdir chatterbox_output
if not exist "speaker_reference" mkdir speaker_reference
if not exist "outputs" mkdir outputs
if not exist "logs" mkdir logs
echo [OK] Directories created.
echo.

REM --- Download pretrained models ---
echo [*] Downloading Chatterbox pretrained models...
echo     (First run downloads ~1.5GB of model files)
pushd chatterbox-finetuning
..\\.venv\Scripts\python.exe setup.py
popd
echo.

REM --- Verify GPU ---
echo [*] Checking GPU availability...
.venv\Scripts\python.exe -c "import torch; gpu=torch.cuda.is_available(); name=torch.cuda.get_device_name(0) if gpu else 'N/A'; vram=torch.cuda.get_device_properties(0).total_mem/(1024**3) if gpu else 0; print(f'  GPU: {gpu}'); print(f'  Device: {name}'); print(f'  VRAM: {vram:.1f} GB')"
echo.

REM --- Vocab size reminder ---
echo ============================================================
echo    IMPORTANT: After setup.py runs, note the vocab size
echo    it outputs (e.g. 52260). This must be set in:
echo      chatterbox-finetuning\src\config.py  (new_vocab_size)
echo      chatterbox-finetuning\inference.py   (NEW_VOCAB_SIZE)
echo ============================================================
echo.

REM --- Done ---
echo ============================================================
echo    Setup Complete!
echo ============================================================
echo.
echo    Next steps:
echo    1. Place voice recordings in:  data\raw\
echo    2. Activate venv:  .venv\Scripts\activate
echo    3. Preprocess:     python scripts\preprocess.py
echo    4. Transcribe:     python scripts\transcribe.py
echo    5. Prepare data:   python scripts\prepare_dataset.py
echo    6. Train:          python scripts\train.py
echo    7. Generate:       python scripts\inference.py --text "Hello"
echo.
echo    Quick zero-shot (no training):
echo       python zero-shot\generate.py --text "Hello world"
echo.
pause
