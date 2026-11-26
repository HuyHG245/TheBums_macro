@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" call setup.bat
if not exist ".venv\Scripts\python.exe" goto failed
".venv\Scripts\python.exe" -c "import cv2, numpy, mss, pydirectinput, keyboard" >nul 2>&1
if errorlevel 1 (
    echo Installing missing macro dependencies in this virtual environment...
    ".venv\Scripts\python.exe" -m pip install -r "%~dp0requirements.txt"
    if errorlevel 1 goto failed
    ".venv\Scripts\python.exe" -c "import cv2, numpy, mss, pydirectinput, keyboard"
    if errorlevel 1 goto failed
)
".venv\Scripts\python.exe" -c "import torch" >nul 2>&1
if errorlevel 1 (
    ".venv\Scripts\python.exe" -m pip install "torch>=2.4" --index-url https://download.pytorch.org/whl/cpu
    if errorlevel 1 goto failed
)
".venv\Scripts\python.exe" -c "from grid114.video_v114 import Reader; Reader()"
if errorlevel 1 goto failed
set "THEBUMS_GRID114=1"
".venv\Scripts\python.exe" main.py
if errorlevel 1 goto failed
exit /b 0
:failed
echo Grid v11.4 could not start. See the message above.
pause
exit /b 1
