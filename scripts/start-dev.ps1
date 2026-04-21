$ErrorActionPreference = "Stop"

# This script opens two dedicated PowerShell windows so frontend and backend
# logs remain visible independently during local development.
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$BackendScript = Join-Path $PSScriptRoot "start-backend.ps1"
$FrontendScript = Join-Path $PSScriptRoot "start-frontend.ps1"

if (-not (Test-Path $BackendScript)) {
    throw "Backend startup script was not found at $BackendScript"
}

if (-not (Test-Path $FrontendScript)) {
    throw "Frontend startup script was not found at $FrontendScript"
}

Write-Host "Launching backend and frontend development windows..." -ForegroundColor Cyan

Start-Process powershell -ArgumentList @(
    "-NoExit",
    "-ExecutionPolicy",
    "Bypass",
    "-File",
    $BackendScript
) -WorkingDirectory $ProjectRoot

Start-Process powershell -ArgumentList @(
    "-NoExit",
    "-ExecutionPolicy",
    "Bypass",
    "-File",
    $FrontendScript
) -WorkingDirectory $ProjectRoot

for ($attempt = 0; $attempt -lt 20; $attempt += 1) {
    Start-Sleep -Milliseconds 500
    try {
        Invoke-WebRequest "http://127.0.0.1:5173" -UseBasicParsing -TimeoutSec 2 | Out-Null
        Start-Process "http://127.0.0.1:5173"
        break
    } catch {
        if ($attempt -eq 19) {
            Write-Host "Frontend did not answer before browser auto-open timeout." -ForegroundColor Yellow
        }
    }
}

Write-Host "Backend:  http://127.0.0.1:8000" -ForegroundColor Green
Write-Host "Frontend: http://127.0.0.1:5173" -ForegroundColor Green
Write-Host "Browser:  http://127.0.0.1:5173" -ForegroundColor Green
