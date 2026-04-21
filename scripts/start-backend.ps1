$ErrorActionPreference = "Stop"

# Resolve the project root relative to this script so the startup flow keeps
# working even if the current shell location changes.
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$BackendRoot = Join-Path $ProjectRoot "apps\backend"
$PythonExe = Join-Path $ProjectRoot ".condaenv\python.exe"
$PortGuardScript = Join-Path $PSScriptRoot "port-guard.ps1"

if (-not (Test-Path $PythonExe)) {
    throw "Project Python environment was not found at $PythonExe"
}

if (-not (Test-Path $PortGuardScript)) {
    throw "Port guard script was not found at $PortGuardScript"
}

. $PortGuardScript

Set-Location $BackendRoot

Write-Host "Starting backend from $BackendRoot" -ForegroundColor Cyan
Write-Host "Using Python executable $PythonExe" -ForegroundColor Cyan

Ensure-RiverPortAvailable -Port 8000 -ProjectRoot $ProjectRoot -ExpectedProcessNames @("python", "python.exe") -ServiceLabel "backend"

& $PythonExe -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
