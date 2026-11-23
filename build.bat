@echo off
setlocal
cd /d "%~dp0"
title The Bums Macro - build
echo ==== The Bums Macro - build ====
echo.

rem Requires Administrator elevation to interact with an elevated Roblox window.
rem Windows will show a UAC consent prompt on every new launch; this is NOT an elevation bypass.
set "ADMIN_FLAG=--uac-admin"

rem Default: ONE file, TheBumsMacro.exe. Run "build.bat onedir" to get a folder instead:
rem it starts faster and antivirus programs complain less about it.
set "MODE=--onefile"
if /i "%~1"=="onedir" set "MODE=--onedir"
rem A validated shared model is automatically included, with CPU runtime.
rem No validated model: Standard stays lean; never ship a dummy model.
set "HYBRID_BUILD=0"
set "HYBRID_FORCE=0"
if /i "%~1"=="hybrid" set "HYBRID_FORCE=1"
if /i "%~2"=="hybrid" set "HYBRID_FORCE=1"

for %%F in (main.py requirements.txt version_info.txt image\thebums.ico updater.ps1 thebums_reconnect.py thebums_features.py thebums_hybrid_ai.py) do (
    if not exist "%%F" (
        echo Missing file: %%F
        goto :fail
    )
)
if not exist "image\" (
    echo The "image" folder was not found next to build.bat
    goto :fail
)
if not exist "guides\" (
    echo The "guides" folder was not found next to build.bat
    goto :fail
)

set "PYTHON=python"
%PYTHON% --version >nul 2>nul
if errorlevel 1 set "PYTHON=py -3"
%PYTHON% --version >nul 2>nul
if errorlevel 1 (
    echo Python was not found. Install Python 3.10 or newer from python.org
    echo and tick "Add Python to PATH" in the installer, then run this file again.
    goto :fail
)

if not exist ".buildenv\Scripts\python.exe" (
    echo Creating the build environment...
    %PYTHON% -m venv .buildenv
    if errorlevel 1 goto :fail
)
set "PY=.buildenv\Scripts\python.exe"

echo Installing the requirements...
"%PY%" -m pip install pyinstaller -r requirements.txt
if errorlevel 1 (
    echo Could not install the requirements. Check your internet connection.
    goto :fail
)
if exist "models\hybrid_manifest.json" (
    echo Verifying shared model and independent held-out report...
    "%PY%" developer\check_hybrid_model.py --models models
    if errorlevel 1 goto :fail
    set "HYBRID_BUILD=1"
)
if "%HYBRID_FORCE%"=="1" if "%HYBRID_BUILD%"=="0" (
    echo Cannot build Hybrid AI: no independently validated shared ONNX model.
    goto :fail
)
set "HYBRID_ARGS="
if "%HYBRID_BUILD%"=="1" (
    echo Installing optional local CPU ONNX runtime for Hybrid AI Beta...
    "%PY%" -m pip install -r requirements-hybrid-ai.txt
    if errorlevel 1 goto :fail
    set "HYBRID_ARGS=--collect-all onnxruntime --add-data models;models"
)

rem Embed a source revision when this is a real Git checkout; otherwise mark unknown.
echo unpublished-local-build>"build_revision.txt"
where git >nul 2>nul
if not errorlevel 1 (
    for /f "delims=" %%H in ('git rev-parse --verify HEAD 2^>nul') do echo %%H>"build_revision.txt"
)
echo.
echo Building - this takes a few minutes...
"%PY%" -m PyInstaller --noconfirm --clean %MODE% --windowed --noupx %ADMIN_FLAG% %HYBRID_ARGS% ^
    --name TheBumsMacro ^
    --icon "image\thebums.ico" ^
    --version-file "version_info.txt" ^
    --add-data "image;image" ^
    --add-data "build_revision.txt;." ^
    --add-data "guides;guides" ^
    --add-data "updater.ps1;." ^
    main.py
if errorlevel 1 (
    if exist "TheBumsMacro.spec" del /q "TheBumsMacro.spec"
    goto :fail
)

if exist "build" rmdir /s /q "build"
if exist "TheBumsMacro.spec" del /q "TheBumsMacro.spec"

echo.
if /i "%~1"=="onedir" (
    echo Done. Run dist\TheBumsMacro\TheBumsMacro.exe - keep that whole folder together.
) else (
    copy /y "dist\TheBumsMacro.exe" "TheBumsMacro.exe" >nul
    echo Done. Your program is TheBumsMacro.exe in this folder, a copy is in dist\.
)
echo There is no console window behind it. Settings and stats are saved in Documents\The Bums.
echo.
pause
exit /b 0

:fail
echo.
echo Build failed - read the messages above.
pause
exit /b 1
