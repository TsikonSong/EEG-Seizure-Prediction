# Evaluation design shapes reported performance in EEG seizure prediction

This repository contains the preprocessing, model, and analysis code for a
benchmark comparing exact-sample overlap, within-case evaluation, and
cold-start testing on unseen subject groups. It also contains the PSD+LDA
external-transfer analysis from CHB-MIT to Siena.

The repository is research software, not a clinical seizure-warning system.
`FPD_300/h` denotes cadence-adjusted false-positive window decisions per
nominal hour. It is not a continuous-stream alarm rate because no smoothing,
merging, refractory period, or time-in-warning policy is applied.

## Reproducibility record

The publication-facing analysis uses 20 fixed splits of 22 subject groups.
CHB-MIT case identifiers `chb01` and `chb21` correspond to one individual and
are always assigned to the same train, validation, or test partition.

Small audit artifacts are included in `results/`:

- the exact CSV tables used by the LaTeX figures and numerical summaries;
- 100 compressed held-out prediction archives (five models x 20 seeds);
- no raw EEG, private clinical information, or model checkpoints.

The following CPU-only check validates every split, verifies that each
prediction archive contains the expected held-out cases, regenerates the
low-FPD table, and matches it to the checked-in manuscript source data:

```bash
python -m pip install -r requirements-analysis.txt
python -m unittest discover -s tests -v
```

On a typical laptop this verification takes under one minute. The same checks
run automatically in GitHub Actions.

## Recreate the low-FPD source table

This command needs only the small prediction archives already in the
repository; it does not require raw EEG or model retraining:

```bash
python run.py scripts/analysis/work_J_far_constrained_sensitivity.py \
  --fpd-ceiling 0.2 \
  --source-data-out outputs/strict_low_far_per_seed.csv
```

The command selects thresholds post hoc on each strict held-out test score
vector. Its output is a descriptive score-separation diagnostic and must not be
interpreted as a locked deployment threshold.

## Software environment

The experiments used Python 3.10. Create the reference environment with either
conda or venv:

```bash
conda env create -f environment.yml
conda activate eeg-seizure-benchmark
```

or

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install a PyTorch build compatible with the local CUDA driver when using a GPU.
Strict training defaults to deterministic PyTorch operations where supported;
cross-platform floating-point results can still differ at the final decimal
place. The checked-in prediction arrays and source tables are the fixed audit
record.

## Public datasets

Raw data are not redistributed here. Download the original releases from
PhysioNet:

- [CHB-MIT Scalp EEG Database v1.0.0](https://physionet.org/content/chbmit/1.0.0/), DOI [10.13026/C2K01R](https://doi.org/10.13026/C2K01R)
- [Siena Scalp EEG Database v1.0.0](https://physionet.org/content/siena-scalp-eeg/1.0.0/), DOI [10.13026/5d4a-j060](https://doi.org/10.13026/5d4a-j060)

CHB-MIT is approximately 42.6 GB uncompressed. See `docs/data_notes.md` for the
window definition, expected folders, and output layout.

## End-to-end strict subject-group workflow

All core scripts accept paths at the command line. Examples below use generic
local folders rather than author-specific drive paths.

1. Preprocess CHB-MIT:

   ```bash
   python run.py scripts/preprocessing/preprocess_chbmit.py \
     --data-dir /path/to/chbmit-1.0.0 \
     --out-dir /path/to/chbmit_preprocessed \
     --temp-dir /path/to/chbmit_temp
   ```

2. Train the five strict subject-grouped models and export held-out prediction
   archives:

   ```bash
   python run.py scripts/analysis/work_H_subject_level_pi.py \
     --models all \
     --data-dir /path/to/chbmit_preprocessed \
     --results-root /path/to/seizure_results
   ```

   Outputs are written beneath
   `/path/to/seizure_results/subject_level_pi/`, including per-seed JSON files,
   model checkpoints, the exact split audit, and compact prediction archives.
   Full deep-model training is GPU-intensive and can take hours depending on
   hardware. Jobs can be resumed because completed seeds are detected.

3. Recreate the low-FPD table from newly generated predictions:

   ```bash
   python run.py scripts/analysis/work_J_far_constrained_sensitivity.py \
     --predictions-root /path/to/seizure_results/subject_level_pi/predictions \
     --out-dir /path/to/seizure_results/analysis_outputs/low_fpd
   ```

4. Preprocess Siena and rerun the strict PSD+LDA transfer:

   ```bash
   python run.py scripts/siena/preprocess_siena.py \
     --data-dir /path/to/siena-scalp-eeg-1.0.0 \
     --out-dir /path/to/siena_preprocessed \
     --temp-dir /path/to/siena_temp

   python run.py scripts/siena/work_L_siena_external_psd_lda.py \
     --chb-dir /path/to/chbmit_preprocessed \
     --siena-dir /path/to/siena_preprocessed \
     --out-dir /path/to/seizure_results/siena_strict_psd_lda
   ```

The Siena script fits PSD+LDA only on the corresponding CHB-MIT training
groups, selects thresholds only on CHB-MIT validation groups, and applies the
model unchanged to the 13 eligible Siena participants.

## Repository layout

```text
src/                     split logic, data loaders, metrics, model definitions
scripts/preprocessing/   CHB-MIT preprocessing and validation
scripts/training/        supporting training and sensitivity scripts
scripts/analysis/        benchmark and source-data analyses
scripts/siena/           Siena preprocessing and PSD+LDA transfer
results/                 curated prediction arrays and manuscript source data
tests/                   split and artifact reproducibility checks
notebooks/               interactive records of earlier analyses
runners/windows/         optional Windows runners for long jobs
docs/                    data and script documentation
```

## Citation and license

Citation metadata are provided in `CITATION.cff`. The code is released under
the MIT License. The original datasets remain governed by their respective
PhysioNet licenses and citation requirements.
