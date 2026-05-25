# Script map

## Shared modules

`src\data_utils.py` defines the CHB-MIT case list, patient splits, PyTorch datasets, and dataloaders.

`src\eval_utils.py` contains threshold selection, AUC, sensitivity, specificity, precision, false-alarm rate, and event-level sensitivity.

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

`scripts\analysis\work_H_subject_level_pi.py` binds chb01 and chb21 to the same subject group.

`scripts\analysis\work_I_far_silencing_simulation.py` studies detector silencing under low false-alarm targets.

`scripts\analysis\work_J_far_constrained_sensitivity.py` reports sensitivity at a fixed false-alarm ceiling.

## Siena

`scripts\siena\work_K_siena_feasibility.py` checks the Siena download and counts candidate windows without full preprocessing.

`scripts\siena\preprocess_siena.py` converts Siena EDF files into CHB-MIT-style window arrays.

`scripts\siena\work_L_siena_external_psd_lda.py` trains PSD+LDA on CHB-MIT splits and evaluates directly on Siena.
