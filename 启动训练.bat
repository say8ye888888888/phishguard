@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Please create the project virtual environment first. See README.md.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" download_pretrained.py
if errorlevel 1 exit /b 1
".venv\Scripts\python.exe" -u train_transformer.py --model .cache/distilbert-base-uncased --lengths 128 --epochs 2 --batch-size 32 --threads 6
pause
