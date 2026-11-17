@echo off
setlocal
cd /d "%~dp0"
title The Bums - build
echo ==== The Bums - build ====
echo.

rem Set to --uac-admin only if the macro must control windows that run as administrator.
rem Leave it empty for fewer antivirus warnings.
set "ADMIN_FLAG="

rem Default: ONE file, TheBums.exe. Run "build.bat onedir" to get a folder instead:
rem it starts faster and antivirus programs complain less about it.
set "MODE=--onefile"
if /i "%~1"=="onedir" set "MODE=--onedir"

for %%F in (main.py requirements.txt change.md version_info.txt image\thebums.ico) do (
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

echo.
echo Building - this takes a few minutes...
"%PY%" -m PyInstaller --noconfirm --clean %MODE% --windowed --noupx %ADMIN_FLAG% ^
    --name TheBums ^
    --icon "image\thebums.ico" ^
    --version-file "version_info.txt" ^
    --add-data "image;image" ^
    --add-data "change.md;." ^
    --add-data "guides;guides" ^
    main.py
if errorlevel 1 goto :fail

if exist "build" rmdir /s /q "build"
if exist "TheBums.spec" del /q "TheBums.spec"

echo.
if /i "%~1"=="onedir" (
    echo Done. Run dist\TheBums\TheBums.exe - keep that whole folder together.
) else (
    copy /y "dist\TheBums.exe" "TheBums.exe" >nul
    echo Done. Your program is TheBums.exe in this folder, a copy is in dist\.
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
