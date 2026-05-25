$ErrorActionPreference = "Continue"

$OutRoot = "D:\seizure_results\subject_level_pi"
$Log = Join-Path $OutRoot "overnight_keep_awake.log"
$Models = @("tcn", "1dcnn", "eeg_conformer")
$Deadline = (Get-Date).AddHours(24)

function Write-KeepAwakeLog {
    param([string]$Message)
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -LiteralPath $Log -Value "[$stamp] $Message"
}

function Test-ModelComplete {
    param([string]$Model)
    $summaryPath = Join-Path (Join-Path $OutRoot $Model) "summary.json"
    if (-not (Test-Path -LiteralPath $summaryPath)) {
        return $false
    }
    try {
        $summary = Get-Content -LiteralPath $summaryPath -Raw | ConvertFrom-Json
        return [int]$summary.n_seeds -ge 20
    }
    catch {
        return $false
    }
}

Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;

public static class SleepBlocker {
    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern uint SetThreadExecutionState(uint esFlags);
}
"@

$ES_CONTINUOUS = [uint32]"0x80000000"
$ES_SYSTEM_REQUIRED = [uint32]"0x00000001"

New-Item -ItemType Directory -Force -Path $OutRoot | Out-Null
Write-KeepAwakeLog "Keep-awake guard started."

while ((Get-Date) -lt $Deadline) {
    $allComplete = $true
    foreach ($model in $Models) {
        if (-not (Test-ModelComplete -Model $model)) {
            $allComplete = $false
            break
        }
    }

    if ($allComplete) {
        Write-KeepAwakeLog "All queued models complete; clearing keep-awake guard."
        [SleepBlocker]::SetThreadExecutionState($ES_CONTINUOUS) | Out-Null
        exit 0
    }

    [SleepBlocker]::SetThreadExecutionState($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED) | Out-Null
    Start-Sleep -Seconds 60
}

Write-KeepAwakeLog "Deadline reached; clearing keep-awake guard."
[SleepBlocker]::SetThreadExecutionState($ES_CONTINUOUS) | Out-Null
