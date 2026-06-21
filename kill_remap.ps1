[CmdletBinding()]
param()

$ErrorActionPreference = "SilentlyContinue"

$RootDir = (Resolve-Path -LiteralPath $PSScriptRoot).Path.TrimEnd("\")
$CurrentPid = $PID
$ExternalToolNames = @(
    "colmap.exe",
    "ffmpeg.exe",
    "glomap.exe",
    "remap-desktop.exe"
)
$ProjectScopedNames = @(
    "cargo.exe",
    "cmd.exe",
    "esbuild.exe",
    "node.exe",
    "npm.exe",
    "npx.exe",
    "powershell.exe",
    "pwsh.exe",
    "py.exe",
    "python.exe",
    "pythonw.exe",
    "rustc.exe",
    "tauri.exe"
)
$RootScopedNames = @(
    "cargo.exe",
    "esbuild.exe",
    "node.exe",
    "npm.exe",
    "npx.exe",
    "py.exe",
    "python.exe",
    "pythonw.exe",
    "rustc.exe",
    "tauri.exe"
)
$ProjectScriptFragments = @(
    "ReMap-GUI.py",
    "desktop_backend.py",
    "remap_server.py",
    "sfm_runner.py",
    "stray_to_colmap.py"
)
$DevToolFragments = @(
    "npm.cmd run desktop:dev",
    "tauri dev",
    "vite"
)

function Get-ProcessSnapshot {
    Get-CimInstance Win32_Process
}

function Get-ExcludedProcessIds {
    $excluded = @{}
    $process = Get-CimInstance Win32_Process -Filter "ProcessId = $CurrentPid"
    $excluded[[int]$CurrentPid] = $true

    while ($process -and $process.ParentProcessId) {
        $parentId = [int]$process.ParentProcessId
        if ($excluded.ContainsKey($parentId)) {
            break
        }
        $excluded[$parentId] = $true
        $process = Get-CimInstance Win32_Process -Filter "ProcessId = $parentId"
    }

    return $excluded
}

function Test-IsReMapProcess {
    param($Process)

    $name = [string]$Process.Name
    $commandLine = [string]$Process.CommandLine
    $executablePath = [string]$Process.ExecutablePath

    if (
        $commandLine.IndexOf("kill_remap.bat", [System.StringComparison]::OrdinalIgnoreCase) -ge 0 -or
        $commandLine.IndexOf("kill_remap.ps1", [System.StringComparison]::OrdinalIgnoreCase) -ge 0
    ) {
        return $false
    }

    if ($ExternalToolNames -contains $name) {
        return $true
    }

    if (($RootScopedNames -contains $name) -and $executablePath.StartsWith($RootDir, [System.StringComparison]::OrdinalIgnoreCase)) {
        return $true
    }

    if (($RootScopedNames -contains $name) -and $commandLine.IndexOf($RootDir, [System.StringComparison]::OrdinalIgnoreCase) -ge 0) {
        return $true
    }

    foreach ($fragment in $ProjectScriptFragments) {
        if (($ProjectScopedNames -contains $name) -and $commandLine.IndexOf($fragment, [System.StringComparison]::OrdinalIgnoreCase) -ge 0) {
            return $true
        }
    }

    foreach ($fragment in $DevToolFragments) {
        if (($ProjectScopedNames -contains $name) -and $commandLine.IndexOf($fragment, [System.StringComparison]::OrdinalIgnoreCase) -ge 0) {
            return $true
        }
    }

    return $false
}

function Get-DescendantProcessIds {
    param(
        [int[]]$SeedIds,
        $Snapshot
    )

    $ids = @{}
    foreach ($id in $SeedIds) {
        $ids[[int]$id] = $true
    }

    $changed = $true
    while ($changed) {
        $changed = $false
        foreach ($process in $Snapshot) {
            $pidValue = [int]$process.ProcessId
            $parentId = [int]$process.ParentProcessId
            if ($ids.ContainsKey($parentId) -and -not $ids.ContainsKey($pidValue)) {
                $ids[$pidValue] = $true
                $changed = $true
            }
        }
    }

    return @($ids.Keys)
}

function Get-Depth {
    param(
        [int]$ProcessId,
        $ById
    )

    $depth = 0
    $seen = @{}
    $cursor = $ProcessId
    while ($ById.ContainsKey($cursor)) {
        if ($seen.ContainsKey($cursor)) {
            break
        }
        $seen[$cursor] = $true
        $parentId = [int]$ById[$cursor].ParentProcessId
        if (-not $ById.ContainsKey($parentId)) {
            break
        }
        $depth += 1
        $cursor = $parentId
    }
    return $depth
}

function Get-ReMapTargets {
    param(
        $Snapshot,
        $ExcludedIds
    )

    $seedIds = @(
        $Snapshot |
            Where-Object { Test-IsReMapProcess $_ } |
            Where-Object { -not $ExcludedIds.ContainsKey([int]$_.ProcessId) } |
            ForEach-Object { [int]$_.ProcessId }
    )

    if ($seedIds.Count -eq 0) {
        return @()
    }

    $targetIds = Get-DescendantProcessIds -SeedIds $seedIds -Snapshot $Snapshot
    return @(
        $Snapshot |
            Where-Object { $targetIds -contains [int]$_.ProcessId } |
            Where-Object { -not $ExcludedIds.ContainsKey([int]$_.ProcessId) }
    )
}

Write-Host "Killing ReMap processes..."

$excludedIds = Get-ExcludedProcessIds
$snapshot = Get-ProcessSnapshot
$targets = Get-ReMapTargets -Snapshot $snapshot -ExcludedIds $excludedIds

if ($targets.Count -eq 0) {
    Write-Host "No ReMap processes were running."
    exit 0
}

$byId = @{}
foreach ($process in $snapshot) {
    $byId[[int]$process.ProcessId] = $process
}

$targets |
    Sort-Object @{ Expression = { Get-Depth -ProcessId ([int]$_.ProcessId) -ById $byId }; Descending = $true } |
    ForEach-Object {
        Stop-Process -Id ([int]$_.ProcessId) -Force
    }

Start-Sleep -Milliseconds 800

$remainingSnapshot = Get-ProcessSnapshot
$remaining = Get-ReMapTargets -Snapshot $remainingSnapshot -ExcludedIds $excludedIds

if ($remaining.Count -gt 0) {
    Write-Host "Some ReMap processes are still running:"
    $remaining |
        Select-Object ProcessId, ParentProcessId, Name, ExecutablePath, CommandLine |
        Format-Table -AutoSize
    exit 1
}

Write-Host "All ReMap processes have been terminated."
exit 0
