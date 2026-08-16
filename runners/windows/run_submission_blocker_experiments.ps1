$ErrorActionPreference = "Continue"

$Python = if ($env:SEIZURE_PYTHON) { $env:SEIZURE_PYTHON } else { "python" }
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$AnalysisScripts = Join-Path $RepoRoot "scripts\analysis"
$ResultsRoot = if ($env:SEIZURE_RESULTS_DIR) { $env:SEIZURE_RESULTS_DIR } else { "D:\seizure_results" }
$SubjectRoot = Join-Path $ResultsRoot "subject_level_pi"
$AnalysisRoot = Join-Path $ResultsRoot "analysis_outputs"
$RunRoot = Join-Path $AnalysisRoot "sensitivity_check_runs"
$ControllerLog = Join-Path $RunRoot "controller.log"
$env:PYTHONPATH = @(
    (Join-Path $RepoRoot "src"),
    (Join-Path $RepoRoot "scripts\analysis"),
    (Join-Path $RepoRoot "scripts\siena"),
    (Join-Path $RepoRoot "scripts\preprocessing"),
    (Join-Path $RepoRoot "scripts\training"),
    $env:PYTHONPATH
) -join ";"

function Write-RunLog {
    param([string]$Message)
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -LiteralPath $ControllerLog -Value "[$stamp] $Message"
}

function Invoke-LoggedCommand {
    param(
        [string]$Name,
        [string]$LogPath,
        [scriptblock]$Command
    )
    Write-RunLog "Starting $Name."
    Push-Location $RepoRoot
    try {
        $env:PYTHONUNBUFFERED = "1"
        & $Command *>> $LogPath
        $exitCode = $LASTEXITCODE
    }
    catch {
        $exitCode = 999
        Add-Content -LiteralPath $LogPath -Value ("Controller caught exception: {0}" -f $_.Exception.Message)
    }
    finally {
        Pop-Location
    }
    Write-RunLog ("Finished {0} with exit code {1}." -f $Name, $exitCode)
    return $exitCode
}

New-Item -ItemType Directory -Force -Path $RunRoot | Out-Null
Write-RunLog "Sensitivity-check experiment controller started."

$psPlan = @(
    @{ Model = "1dcnn"; OutDir = Join-Path $AnalysisRoot "work_G_ps_leakage_1dcnn"; Log = Join-Path $RunRoot "work_G_ps_leakage_1dcnn.log" },
    @{ Model = "eeg_conformer"; OutDir = Join-Path $AnalysisRoot "work_G_ps_leakage_eeg_conformer"; Log = Join-Path $RunRoot "work_G_ps_leakage_eeg_conformer.log" }
)

foreach ($item in $psPlan) {
    $summaryPath = Join-Path $item.OutDir "ps_split_summary.csv"
    if (Test-Path -LiteralPath $summaryPath) {
        Write-RunLog ("PS leakage {0} summary already exists; skipping." -f $item.Model)
        continue
    }
    New-Item -ItemType Directory -Force -Path $item.OutDir | Out-Null
    Invoke-LoggedCommand -Name ("work_G_ps_leakage_" + $item.Model) -LogPath $item.Log -Command {
        & $Python -u (Join-Path $AnalysisScripts "work_G_ps_leakage_audit.py") --models $item.Model --n-random 20 --max-epochs 100 --patience 20 --out-dir $item.OutDir
    } | Out-Null
}

$lowFpdOut = Join-Path $AnalysisRoot "work_J_low_fpd"
$lowFpdSummary = Join-Path $lowFpdOut "strict_low_fpd_summary.csv"
if (Test-Path -LiteralPath $lowFpdSummary) {
    Write-RunLog "Low-FPD sensitivity summary already exists; skipping."
}
else {
    Invoke-LoggedCommand -Name "work_J_low_fpd" -LogPath (Join-Path $RunRoot "work_J_low_fpd.log") -Command {
        & $Python -u (Join-Path $AnalysisScripts "work_J_far_constrained_sensitivity.py") --fpd-ceiling 0.2 --predictions-root (Join-Path $SubjectRoot "predictions") --out-dir $lowFpdOut
    } | Out-Null
}

Write-RunLog "Sensitivity-check experiment controller completed queued work."
