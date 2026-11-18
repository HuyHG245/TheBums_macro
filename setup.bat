@echo off
setlocal
cd /d "%~dp0"
set "PYTHON=python"
%PYTHON% --version >nul 2>nul
if errorlevel 1 set "PYTHON=py -3"
%PYTHON% --version >nul 2>nul
if errorlevel 1 (
    echo Python was not found. Install Python 3.10 or newer from python.org
    echo and tick "Add Python to PATH" in the installer, then run this file again.
    pause
    exit /b 1
)
if not exist ".venv\Scripts\python.exe" %PYTHON% -m venv .venv
if errorlevel 1 (
    echo Could not create the virtual environment.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
    echo Could not install the requirements. Check your internet connection.
    pause
    exit /b 1
)
echo.
echo Done. Double-click run.bat to start The Bums Macro.
pause
