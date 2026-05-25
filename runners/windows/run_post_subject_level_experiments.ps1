$ErrorActionPreference = "Continue"

$Python = if ($env:SEIZURE_PYTHON) { $env:SEIZURE_PYTHON } else { "python" }
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$AnalysisScripts = Join-Path $RepoRoot "scripts\analysis"
$SubjectRoot = "D:\seizure_results\subject_level_pi"
$AnalysisRoot = "D:\seizure_results\analysis_outputs"
$ControllerLog = Join-Path $SubjectRoot "post_subject_controller.log"
$SubjectModels = @("tcn", "1dcnn", "eeg_conformer")
$env:PYTHONPATH = @(
    (Join-Path $RepoRoot "src"),
    (Join-Path $RepoRoot "scripts\analysis"),
    (Join-Path $RepoRoot "scripts\siena"),
    (Join-Path $RepoRoot "scripts\preprocessing"),
    (Join-Path $RepoRoot "scripts\training"),
    $env:PYTHONPATH
) -join ";"

function Write-PostLog {
    param([string]$Message)
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -LiteralPath $ControllerLog -Value "[$stamp] $Message"
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

function Wait-SubjectQueue {
    Write-PostLog "Waiting for subject-level PI queue to complete."
    while ($true) {
        $complete = $true
        foreach ($model in $SubjectModels) {
            if (-not (Test-SubjectModelComplete -Model $model)) {
                $complete = $false
                $count = @(Get-ChildItem -LiteralPath (Join-Path $SubjectRoot $model) -Filter "seed*_result.json" -ErrorAction SilentlyContinue).Count
                Write-PostLog ("Subject-level {0} incomplete; seed_result_files={1}" -f $model, $count)
            }
        }
        if ($complete) {
            Write-PostLog "All queued subject-level PI models complete."
            return
        }
        Start-Sleep -Seconds 600
    }
}

function Invoke-LoggedCommand {
    param(
        [string]$Name,
        [string]$LogPath,
        [scriptblock]$Command
    )
    Write-PostLog "Starting $Name."
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
    Write-PostLog ("Finished {0} with exit code {1}." -f $Name, $exitCode)
    return $exitCode
}

New-Item -ItemType Directory -Force -Path $SubjectRoot | Out-Null
Write-PostLog "Post-subject experiment controller started."
Wait-SubjectQueue

$farSummary = Join-Path $AnalysisRoot "work_I_far_silencing\far_silencing_summary.csv"
if (Test-Path -LiteralPath $farSummary) {
    Write-PostLog "FAR silencing simulation already complete; skipping."
}
else {
    Invoke-LoggedCommand -Name "work_I_far_silencing" -LogPath (Join-Path $SubjectRoot "work_I_far_silencing.log") -Command {
        & $Python -u (Join-Path $AnalysisScripts "work_I_far_silencing_simulation.py")
    } | Out-Null
}

$psPlan = @(
    @{ Model = "eegnet"; OutDir = Join-Path $AnalysisRoot "work_G_ps_leakage_eegnet"; Log = Join-Path $SubjectRoot "work_G_ps_leakage_eegnet.log" },
    @{ Model = "tcn"; OutDir = Join-Path $AnalysisRoot "work_G_ps_leakage_tcn"; Log = Join-Path $SubjectRoot "work_G_ps_leakage_tcn.log" }
)

foreach ($item in $psPlan) {
    $summaryPath = Join-Path $item.OutDir "ps_split_summary.csv"
    if (Test-Path -LiteralPath $summaryPath) {
        Write-PostLog ("PS leakage {0} summary already exists; skipping." -f $item.Model)
        continue
    }
    New-Item -ItemType Directory -Force -Path $item.OutDir | Out-Null
    Invoke-LoggedCommand -Name ("work_G_ps_leakage_" + $item.Model) -LogPath $item.Log -Command {
        & $Python -u (Join-Path $AnalysisScripts "work_G_ps_leakage_audit.py") --models $item.Model --n-random 20 --max-epochs 100 --patience 20 --out-dir $item.OutDir
    } | Out-Null
}

Write-PostLog "Post-subject experiment controller completed queued work."
