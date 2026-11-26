@echo off
setlocal EnableExtensions
cd /d "%~dp0"
if not "%errorlevel%"=="0" goto fail
title The Bums AI Studio - developer only
set "AIENV=%CD%\datasets\ai_studio_env"
set "AIPY=%AIENV%\Scripts\python.exe"
if exist "%AIPY%" goto verify

echo Creating private Python environment. No administrator access required.
where py >nul 2>nul
if not "%errorlevel%"=="0" goto fallback
call :probe_launcher 3.12
if "%errorlevel%"=="0" goto make_env
call :probe_launcher 3.11
if "%errorlevel%"=="0" goto make_env
call :probe_launcher 3.10
if "%errorlevel%"=="0" goto make_env

:fallback
where python >nul 2>nul
if not "%errorlevel%"=="0" goto no_python
python -c "import sys,struct,tkinter,venv; assert (3,10) <= sys.version_info[:2] <= (3,12) and struct.calcsize('P') == 8" >nul 2>nul
if not "%errorlevel%"=="0" goto no_python
echo Using supported Python from PATH.
python -m venv "%AIENV%"
if not "%errorlevel%"=="0" goto env_failed
goto verify

:make_env
echo Using Python %PYVER% from the Python launcher.
py -%PYVER% -m venv "%AIENV%"
if not "%errorlevel%"=="0" goto env_failed

:verify
if not exist "%AIPY%" goto env_failed
"%AIPY%" -c "import sys,struct,tkinter,venv,pip; assert sys.prefix != sys.base_prefix and (3,10) <= sys.version_info[:2] <= (3,12) and struct.calcsize('P') == 8" >nul 2>nul
if not "%errorlevel%"=="0" goto env_failed
"%AIPY%" -c "import cv2, numpy" >nul 2>nul
if "%errorlevel%"=="0" goto start
echo First setup: installing OpenCV and NumPy only, without a download cache.
"%AIPY%" -m pip install --no-cache-dir "numpy>=1.24,<3" "opencv-python>=4.8,<5"
if not "%errorlevel%"=="0" goto fail
"%AIPY%" -c "import cv2, numpy"
if not "%errorlevel%"=="0" goto fail

:start
"%AIPY%" developer\ai_studio.py
if not "%errorlevel%"=="0" goto fail
exit /b 0

:probe_launcher
rem Compare exact exit status: newer Python managers may return negative
rem Windows HRESULTs, which are not detected by IF ERRORLEVEL 1.
py -%~1 -c "import sys,struct,tkinter,venv; assert sys.version_info[:2] == tuple(map(int,'%~1'.split('.'))) and struct.calcsize('P') == 8" >nul 2>nul
if not "%errorlevel%"=="0" exit /b 1
set "PYVER=%~1"
exit /b 0

:no_python
echo No supported 64-bit Python 3.10, 3.11 or 3.12 with Tkinter was found.
echo If you use the new Python install manager, run: py install 3.12
echo Otherwise install 64-bit Python 3.12 with Tkinter from python.org.
goto fail

:env_failed
echo The private Python environment could not be created or verified.
echo Check the error above. If the environment is broken, close Studio and
echo rename only datasets\ai_studio_env, then rerun this launcher.
echo Do not delete the dataset or training reports.
goto fail

:fail
echo AI Studio could not start. Read the error above.
pause
exit /b 1
