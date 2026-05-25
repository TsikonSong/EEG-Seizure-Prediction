$ErrorActionPreference = "Continue"

$SubjectRoot = "D:\seizure_results\subject_level_pi"
$AnalysisRoot = "D:\seizure_results\analysis_outputs"
$LogPath = Join-Path $SubjectRoot "post_subject_keep_awake.log"

Add-Type -Namespace Win32 -Name NativeMethods -MemberDefinition @"
[System.Runtime.InteropServices.DllImport("kernel32.dll", SetLastError = true)]
public static extern uint SetThreadExecutionState(uint esFlags);
"@

$ES_CONTINUOUS = [uint32]"0x80000000"
$ES_SYSTEM_REQUIRED = [uint32]"0x00000001"
$ES_AWAYMODE_REQUIRED = [uint32]"0x00000040"

function Write-KeepAwakeLog {
    param([string]$Message)
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -LiteralPath $LogPath -Value "[$stamp] $Message"
}

function Test-SubjectModelComplete {
    param([string]$Model)
    $summaryPath = Join-Path (Join-Path $SubjectRoot $Model) "summary.json"
    if (-not (Test-Path -LiteralPath $summaryPath)) {
        return $false
    }
    try {
        $summary = Get-Content -LiteralPath $summaryPath -Raw | ConvertFrom-Json
        return ([int]$summary.n_seeds -ge 20)
    }
    catch {
        return $false
    }
}

function Test-AllPostWorkComplete {
    foreach ($model in @("tcn", "1dcnn", "eeg_conformer")) {
        if (-not (Test-SubjectModelComplete -Model $model)) {
            return $false
        }
    }
    if (-not (Test-Path -LiteralPath (Join-Path $AnalysisRoot "work_I_far_silencing\far_silencing_summary.csv"))) {
        return $false
    }
    if (-not (Test-Path -LiteralPath (Join-Path $AnalysisRoot "work_G_ps_leakage_eegnet\ps_split_summary.csv"))) {
        return $false
    }
    if (-not (Test-Path -LiteralPath (Join-Path $AnalysisRoot "work_G_ps_leakage_tcn\ps_split_summary.csv"))) {
        return $false
    }
    return $true
}

Write-KeepAwakeLog "Post-subject keep-awake guard started."
$deadline = (Get-Date).AddHours(48)
while ((Get-Date) -lt $deadline) {
    [Win32.NativeMethods]::SetThreadExecutionState($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED -bor $ES_AWAYMODE_REQUIRED) | Out-Null
    if (Test-AllPostWorkComplete) {
        Write-KeepAwakeLog "All post-subject work complete; releasing keep-awake guard."
        break
    }
    Start-Sleep -Seconds 300
}

[Win32.NativeMethods]::SetThreadExecutionState($ES_CONTINUOUS) | Out-Null
Write-KeepAwakeLog "Post-subject keep-awake guard stopped."
