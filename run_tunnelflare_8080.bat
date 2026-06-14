@echo off
cd /d "%~dp0"

set "CLOUDFLARED=C:\Program Files (x86)\cloudflared\cloudflared.exe"
if exist "%CLOUDFLARED%" goto run_tunnel

where cloudflared >NUL 2>NUL
if not errorlevel 1 (
    set "CLOUDFLARED=cloudflared"
    goto run_tunnel
)

echo cloudflared.exe not found at the default install path.
echo Default path: C:\Program Files (x86)\cloudflared\cloudflared.exe
echo Install cloudflared or add it to PATH.
pause
exit /b 1

:run_tunnel
"%CLOUDFLARED%" tunnel --protocol http2 --url http://127.0.0.1:8080
