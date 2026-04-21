@echo off
setlocal

REM Resolve the project root to the directory containing this batch file.
set "PROJECT_ROOT=%~dp0"

REM Normalize trailing slash handling for clearer echo output.
if "%PROJECT_ROOT:~-1%"=="\" set "PROJECT_ROOT=%PROJECT_ROOT:~0,-1%"

set "BACKEND_SCRIPT=%PROJECT_ROOT%\scripts\start-backend.ps1"
set "FRONTEND_SCRIPT=%PROJECT_ROOT%\scripts\start-frontend.ps1"

if not exist "%BACKEND_SCRIPT%" (
    echo Backend startup script not found: %BACKEND_SCRIPT%
    exit /b 1
)

if not exist "%FRONTEND_SCRIPT%" (
    echo Frontend startup script not found: %FRONTEND_SCRIPT%
    exit /b 1
)

echo Launching River development environment...
echo Project root: %PROJECT_ROOT%

start "River Backend" powershell -NoExit -ExecutionPolicy Bypass -File "%BACKEND_SCRIPT%"
start "River Frontend" powershell -NoExit -ExecutionPolicy Bypass -File "%FRONTEND_SCRIPT%"

powershell -NoProfile -Command "$ready=$false; for ($i=0; $i -lt 20; $i++) { Start-Sleep -Milliseconds 500; try { Invoke-WebRequest 'http://127.0.0.1:5173' -UseBasicParsing -TimeoutSec 2 | Out-Null; $ready=$true; break } catch {} }; if ($ready) { Start-Process 'http://127.0.0.1:5173' } else { Write-Host 'Frontend did not answer before browser auto-open timeout.' -ForegroundColor Yellow }"

echo Backend:  http://127.0.0.1:8000
echo Frontend: http://127.0.0.1:5173
echo Browser:  http://127.0.0.1:5173
echo.
echo Two new terminal windows should now be starting, and the browser should open automatically.

endlocal
