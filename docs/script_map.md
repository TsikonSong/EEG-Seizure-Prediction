# Script map

## Shared modules

`src/splits.py` defines the fixed seeds, legacy case-ID splits, and strict
22-subject-group splits without requiring PyTorch.

`src/data_utils.py` defines the PyTorch datasets and dataloaders and re-exports
the split helpers for compatibility with the original notebooks.

`src/eval_utils.py` contains threshold selection, AUC, sensitivity, specificity,
precision, cadence-adjusted FPD_300, and event-level sensitivity. The legacy
`false_alarm_rate` name is retained only as a compatibility alias.

`src\models.py` contains the 1D-CNN, EEGNet, TCN, and EEG-Conformer definitions used by the notebooks and scripts.

## Preprocessing

`scripts\preprocessing\preprocess_chbmit.py` creates the main CHB-MIT window arrays.

`scripts\preprocessing\preprocess_chbmit_sensitivity.py` and `preprocess_chbmit_wideband.py` create preprocessing variants for sensitivity checks.

`scripts\preprocessing\validate_preprocessing.py` checks channel counts, shapes, labels, and basic window statistics.

## Main analysis

`scripts\analysis\work_A_sens_spec_bimodality.py` summarizes operating-point patterns.

`scripts\analysis\work_B_wideband_analysis.py` compares the wideband preprocessing variant.

`scripts\analysis\work_C_signal_visualization.py` creates signal and spectrum figures.

`scripts\analysis\work_D_case_series.py` creates the case-series figure panels.

`scripts\analysis\work_E_permutation_null.py` runs the permutation-null check.

`scripts\analysis\work_F_exclude_chb19_sensitivity.py` reruns the summary with chb19 excluded.

`scripts\analysis\work_G_ps_leakage_audit.py` compares chronological and random patient-specific splits.

`scripts/analysis/work_H_subject_level_pi.py` runs the primary strict
subject-grouped benchmark, binds `chb01` and `chb21`, and exports compact
held-out prediction archives for every model and seed.

`scripts\analysis\work_I_far_silencing_simulation.py` studies detector silencing under low false-alarm targets.

`scripts/analysis/work_J_far_constrained_sensitivity.py` validates the strict
prediction archives and reports post hoc sensitivity at a fixed FPD_300
ceiling. It also writes the exact manuscript-compatible source table.

## Siena

`scripts\siena\work_K_siena_feasibility.py` checks the Siena download and counts candidate windows without full preprocessing.

`scripts\siena\preprocess_siena.py` converts Siena EDF files into CHB-MIT-style window arrays.

`scripts/siena/work_L_siena_external_psd_lda.py` trains PSD+LDA on strict
CHB-MIT subject-group splits and evaluates the fitted model directly on Siena,
without target-dataset fitting or calibration.
