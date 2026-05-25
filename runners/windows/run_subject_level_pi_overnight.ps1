$ErrorActionPreference = "Continue"

$Python = if ($env:SEIZURE_PYTHON) { $env:SEIZURE_PYTHON } else { "python" }
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Script = Join-Path $RepoRoot "scripts\analysis\work_H_subject_level_pi.py"
$OutRoot = "D:\seizure_results\subject_level_pi"
$ControllerLog = Join-Path $OutRoot "overnight_controller.log"
$Models = @("tcn", "1dcnn", "eeg_conformer")
$env:PYTHONPATH = @(
    (Join-Path $RepoRoot "src"),
    (Join-Path $RepoRoot "scripts\analysis"),
    (Join-Path $RepoRoot "scripts\siena"),
    (Join-Path $RepoRoot "scripts\preprocessing"),
    (Join-Path $RepoRoot "scripts\training"),
    $env:PYTHONPATH
) -join ";"

function Write-ControllerLog {
    param([string]$Message)
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -LiteralPath $ControllerLog -Value "[$stamp] $Message"
}

function Get-SubjectPiProcesses {
    Get-CimInstance Win32_Process |
        Where-Object {
            $_.CommandLine -and
            $_.CommandLine -like "*python*" -and
            $_.CommandLine -like "*work_H_subject_level_pi.py*"
        } |
        Select-Object ProcessId, CommandLine, CreationDate
}

function Get-SeedResultCount {
    param([string]$Model)
    $modelDir = Join-Path $OutRoot $Model
    if (-not (Test-Path -LiteralPath $modelDir)) {
        return 0
    }
    return @(Get-ChildItem -LiteralPath $modelDir -Filter "seed*_result.json" -ErrorAction SilentlyContinue).Count
}

function Log-ModelSummary {
    param([string]$Model)
    $summaryPath = Join-Path (Join-Path $OutRoot $Model) "summary.json"
    $count = Get-SeedResultCount -Model $Model
    if (Test-Path -LiteralPath $summaryPath) {
        try {
            $summary = Get-Content -LiteralPath $summaryPath -Raw | ConvertFrom-Json
            Write-ControllerLog ("{0} summary: n_seeds={1}, auc_mean={2}, auc_sd={3}, far_mean={4}, far_sd={5}, seed_result_files={6}" -f $Model, $summary.n_seeds, $summary.auc_mean, $summary.auc_sd, $summary.far_mean, $summary.far_sd, $count)
        }
        catch {
            Write-ControllerLog ("{0} summary exists but could not be parsed: {1}; seed_result_files={2}" -f $Model, $_.Exception.Message, $count)
        }
    }
    else {
        Write-ControllerLog ("{0} summary not found yet; seed_result_files={1}" -f $Model, $count)
    }
}

New-Item -ItemType Directory -Force -Path $OutRoot | Out-Null
Write-ControllerLog "Overnight subject-level PI controller started."

while ($true) {
    $existing = @(Get-SubjectPiProcesses)
    if ($existing.Count -eq 0) {
        break
    }
    foreach ($proc in $existing) {
        Write-ControllerLog ("Waiting for existing subject-level PI process pid={0}, created={1}, command={2}" -f $proc.ProcessId, $proc.CreationDate, $proc.CommandLine)
    }
    foreach ($model in $Models) {
        Log-ModelSummary -Model $model
    }
    Start-Sleep -Seconds 300
}

foreach ($model in $Models) {
    Write-ControllerLog "Starting model $model."
    Log-ModelSummary -Model $model

    $modelLog = Join-Path $OutRoot ("{0}_run.log" -f $model)
    $env:PYTHONUNBUFFERED = "1"
    Push-Location $RepoRoot
    try {
        & $Python -u $Script --models $model *>> $modelLog
        $exitCode = $LASTEXITCODE
    }
    catch {
        $exitCode = 999
        Add-Content -LiteralPath $modelLog -Value ("Controller caught exception: {0}" -f $_.Exception.Message)
    }
    finally {
        Pop-Location
    }

    Write-ControllerLog ("Finished model {0} with exit code {1}." -f $model, $exitCode)
    Log-ModelSummary -Model $model

    if ($exitCode -ne 0) {
        Write-ControllerLog ("Model {0} exited non-zero; continuing to next queued model so overnight GPU time is still used." -f $model)
    }
}

Write-ControllerLog "Overnight subject-level PI controller completed queued models."
