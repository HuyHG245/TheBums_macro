@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Install the Standard dependencies first with setup.bat.
    pause
    exit /b 1
)
if not exist "models\hybrid_manifest.json" (
    echo No validated AI model shipped with this source build.
    echo Installing ONNX Runtime alone will NOT enable Hybrid AI.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" developer\check_hybrid_model.py --models models
if errorlevel 1 (
    pause
    exit /b 1
)
echo Installing CPU-only ONNX Runtime for verified shared Hybrid AI...
".venv\Scripts\python.exe" -m pip install -r requirements-hybrid-ai.txt
if errorlevel 1 (
    echo Optional Hybrid AI dependency install failed.
    pause
    exit /b 1
)
echo CPU runtime installed. Hybrid AI also needs a locally trained and validated ONNX model.
pause
