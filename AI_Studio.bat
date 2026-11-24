@echo off
setlocal
cd /d "%~dp0"
title The Bums AI Studio - developer only
set "AIENV=%CD%\datasets\ai_studio_env"
set "AIPY=%AIENV%\Scripts\python.exe"
if exist "%AIPY%" goto verify
echo Creating private Python environment. No administrator access required.
where py >nul 2>nul
if errorlevel 1 goto fallback
for %%V in (3.12 3.11 3.10) do (
    py -%%V -c "import tkinter" >nul 2>nul
    if not errorlevel 1 (
        set "PYVER=%%V"
        goto make_env
    )
)
:fallback
python -c "import sys, tkinter; assert (3,10) <= sys.version_info[:2] <= (3,12)" >nul 2>nul
if errorlevel 1 (
    echo Install 64-bit Python 3.11 or 3.12 with Tkinter from python.org.
    goto fail
)
python -m venv "%AIENV%"
if errorlevel 1 goto fail
goto verify
:make_env
py -%PYVER% -m venv "%AIENV%"
if errorlevel 1 goto fail
:verify
"%AIPY%" -c "import tkinter, cv2, numpy" >nul 2>nul
if not errorlevel 1 goto start
echo First setup: installing OpenCV and NumPy only, without a download cache.
"%AIPY%" -m pip install --no-cache-dir "numpy>=1.24,<3" "opencv-python>=4.8,<5"
if errorlevel 1 goto fail
:start
"%AIPY%" developer\ai_studio.py
if errorlevel 1 goto fail
exit /b 0
:fail
echo AI Studio could not start. Read the error above.
pause
exit /b 1
