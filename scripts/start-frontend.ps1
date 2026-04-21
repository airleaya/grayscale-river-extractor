$ErrorActionPreference = "Stop"

# Keep the startup path deterministic so the script works from any shell
# location without relying on the caller's current directory.
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$FrontendRoot = Join-Path $ProjectRoot "apps\frontend"
$PortGuardScript = Join-Path $PSScriptRoot "port-guard.ps1"

if (-not (Test-Path (Join-Path $FrontendRoot "package.json"))) {
    throw "Frontend package.json was not found at $FrontendRoot"
}

if (-not (Test-Path $PortGuardScript)) {
    throw "Port guard script was not found at $PortGuardScript"
}

. $PortGuardScript

Set-Location $FrontendRoot

Write-Host "Starting frontend from $FrontendRoot" -ForegroundColor Cyan

Ensure-RiverPortAvailable -Port 5173 -ProjectRoot $ProjectRoot -ExpectedProcessNames @("node") -ServiceLabel "frontend"

$ViteConfig = Join-Path $FrontendRoot "vite.config.mjs"
$ViteCache = Join-Path $FrontendRoot ".vite-cache"

if (-not (Test-Path $ViteConfig)) {
    throw "Frontend Vite config was not found at $ViteConfig"
}

if (Test-Path $ViteCache) {
    Write-Host "Clearing stale Vite cache at $ViteCache" -ForegroundColor DarkCyan
    Remove-Item -LiteralPath $ViteCache -Recurse -Force -ErrorAction SilentlyContinue
}

npx vite --config $ViteConfig --configLoader native --host 127.0.0.1 --port 5173
