$ErrorActionPreference = "Stop"

function Get-ListeningPortProcessInfo {
    param(
        [Parameter(Mandatory = $true)]
        [int]$Port
    )

    $netstatLines = netstat -ano -p TCP | Select-String -Pattern "LISTENING"
    foreach ($line in $netstatLines) {
        $columns = ($line.ToString() -replace "\s+", " ").Trim().Split(" ")
        if ($columns.Length -lt 5) {
            continue
        }

        $localAddress = $columns[1]
        $processId = $columns[-1]
        if (-not $localAddress.EndsWith(":$Port")) {
            continue
        }
        if (-not [int]::TryParse($processId, [ref]$null)) {
            continue
        }

        try {
            $process = Get-Process -Id ([int]$processId) -ErrorAction Stop
        } catch {
            continue
        }

        $commandLine = $null
        try {
            $cimProcess = Get-CimInstance Win32_Process -Filter "ProcessId = $processId" -ErrorAction Stop
            $commandLine = $cimProcess.CommandLine
        } catch {
            $commandLine = $null
        }

        return [pscustomobject]@{
            Port = $Port
            ProcessId = [int]$processId
            ProcessName = $process.ProcessName
            Path = $process.Path
            CommandLine = $commandLine
        }
    }

    return $null
}

function Test-RiverOwnedProcess {
    param(
        [Parameter(Mandatory = $true)]
        [pscustomobject]$ProcessInfo,
        [Parameter(Mandatory = $true)]
        [string]$ProjectRoot,
        [Parameter(Mandatory = $true)]
        [string[]]$ExpectedProcessNames
    )

    $processNameMatches = $ExpectedProcessNames -contains $ProcessInfo.ProcessName.ToLowerInvariant()
    $pathMatches = $false
    $commandMatches = $false

    if ($ProcessInfo.Path) {
        $pathMatches = $ProcessInfo.Path.StartsWith($ProjectRoot, [System.StringComparison]::OrdinalIgnoreCase)
    }

    if ($ProcessInfo.CommandLine) {
        $commandMatches = $ProcessInfo.CommandLine.IndexOf($ProjectRoot, [System.StringComparison]::OrdinalIgnoreCase) -ge 0
    }

    return $processNameMatches -or $pathMatches -or $commandMatches
}

function Ensure-RiverPortAvailable {
    param(
        [Parameter(Mandatory = $true)]
        [int]$Port,
        [Parameter(Mandatory = $true)]
        [string]$ProjectRoot,
        [Parameter(Mandatory = $true)]
        [string[]]$ExpectedProcessNames,
        [Parameter(Mandatory = $true)]
        [string]$ServiceLabel
    )

    $processInfo = Get-ListeningPortProcessInfo -Port $Port
    if ($null -eq $processInfo) {
        return
    }

    $normalizedNames = $ExpectedProcessNames | ForEach-Object { $_.ToLowerInvariant() }
    if (-not (Test-RiverOwnedProcess -ProcessInfo $processInfo -ProjectRoot $ProjectRoot -ExpectedProcessNames $normalizedNames)) {
        throw "$ServiceLabel port $Port is already occupied by process $($processInfo.ProcessName) (PID $($processInfo.ProcessId)). Please close that process manually and try again."
    }

    Write-Host "Stopping stale $ServiceLabel process $($processInfo.ProcessName) (PID $($processInfo.ProcessId)) on port $Port" -ForegroundColor Yellow
    Stop-Process -Id $processInfo.ProcessId -Force -ErrorAction Stop
    Start-Sleep -Milliseconds 400
}
