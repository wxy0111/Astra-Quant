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

echo Training and validating a candidate Astra-Quant LightGBM model.
echo The live model is promoted only if validation gates pass.
"%PYTHON_EXE%" -m backtest.train_and_promote_model
pause
