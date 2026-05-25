# Seizure prediction benchmark

This repository contains the code used for a patient-independent seizure prediction benchmark on CHB-MIT. It also includes the Siena preparation scripts used for the external-data checks.

The data are not included. The scripts expect the local data and result folders listed below unless you edit the path constants.

## Data locations

The current scripts use these paths:

```text
D:\chbmit_data
D:\chbmit_preprocessed
D:\Siena\siena-scalp-eeg-1.0.0
D:\siena_preprocessed
D:\seizure_results
```

For a new machine, either recreate those folders or update the constants near the top of the relevant scripts.

## Setup

The experiments were run with Python 3.10. A conda environment is the easiest way to keep the EEG and PyTorch dependencies stable.

```powershell
conda create -n seizure_prediction python=3.10
conda activate seizure_prediction
pip install -r requirements.txt
```

Install the PyTorch build that matches your GPU and CUDA version if the default wheel is not suitable.

## Running scripts

Use `run.py` from the repository root. It adds `src` and the script folders to `PYTHONPATH`, so the old flat-file imports still work.

```powershell
cd D:\seizure_prediction_benchmark_github
python run.py scripts\siena\work_K_siena_feasibility.py
python run.py scripts\siena\work_L_siena_external_psd_lda.py
python run.py scripts\analysis\work_J_far_constrained_sensitivity.py --far-ceiling 0.2
```

Long GPU jobs can also be started with the PowerShell runners under `runners\windows`.

## Repository layout

```text
src\                     shared data loaders, metrics, and model definitions
scripts\preprocessing\   CHB-MIT preprocessing and validation
scripts\training\        sensitivity and wideband training scripts
scripts\analysis\        manuscript analysis scripts
scripts\siena\           Siena feasibility, preprocessing, and transfer probe
notebooks\training\      main model notebooks
notebooks\leaky\         random-window baseline notebooks
notebooks\analysis\      interpretability and statistical notebooks
runners\windows\         Windows runners for long jobs
docs\                    script map and data notes
```

## Main analysis groups

`work_A` through `work_J` reproduce the main analysis panels and robustness checks. `work_K` and `work_L` cover the Siena preparation and external-transfer check.

The notebooks record the interactive runs. The reusable code lives in `src` and `scripts`.

## Data policy

Do not commit EDF files, NumPy window arrays, model checkpoints, or result folders. The `.gitignore` file excludes the usual large local artifacts.
