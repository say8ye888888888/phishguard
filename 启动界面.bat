@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Project environment not found. Please follow README.md to create .venv.
    pause
    exit /b 1
)
echo Open http://127.0.0.1:8501 after the server starts.
".venv\Scripts\python.exe" -I -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501 --browser.gatherUsageStats false
pause
