@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Install the Standard dependencies first with setup.bat.
    pause
    exit /b 1
)
echo Installing optional CPU-only ONNX Runtime for Hybrid AI Beta...
".venv\Scripts\python.exe" -m pip install -r requirements-hybrid-ai.txt
if errorlevel 1 (
    echo Optional Hybrid AI dependency install failed.
    pause
    exit /b 1
)
echo CPU runtime installed. Hybrid AI also needs a locally trained and validated ONNX model.
pause
