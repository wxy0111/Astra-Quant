@echo off
cd /d "%~dp0"

set "PYTHON_EXE=.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" (
    where python >NUL 2>NUL
    if errorlevel 1 (
        echo Cannot find Python. Please create .venv or add python to PATH.
        pause
        exit /b 1
    )
    set "PYTHON_EXE=python"
)

echo Starting Cloudflare Tunnel in a separate terminal...
start "OKX New Tunnel 8080" /d "%~dp0" cmd /k call "run_tunnelflare_8080.bat"
echo Copy the trycloudflare.com URL from the tunnel terminal to your phone.

echo Starting OKX New dashboard at http://127.0.0.1:8080
"%PYTHON_EXE%" main.py
pause
