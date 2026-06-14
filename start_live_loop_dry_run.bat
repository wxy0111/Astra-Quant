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

echo Starting OKX New live loop in dry-run mode.
echo No real orders will be sent.
echo Dashboard will run at http://127.0.0.1:8080 in the same process.
echo Cloudflare Tunnel output will appear in this terminal.
"%PYTHON_EXE%" -m backtest.run_live_loop_okx --dashboard --tunnel --dry-run
pause
