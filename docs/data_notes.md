# Data and preprocessing notes

The repository does not redistribute raw EEG or the large preprocessed window
arrays. Small held-out prediction vectors are included for auditing the
publication-facing analyses.

## CHB-MIT

Source: [CHB-MIT Scalp EEG Database v1.0.0](https://physionet.org/content/chbmit/1.0.0/), DOI [10.13026/C2K01R](https://doi.org/10.13026/C2K01R).

The benchmark includes 23 case identifiers (`chb01`--`chb23`) after excluding
`chb24` because the fixed 18-channel montage was unavailable. `chb01` and
`chb21` belong to one individual, giving 22 subject groups.

Main preprocessing parameters:

- 20-second windows at 256 Hz;
- 18 bipolar channels;
- 0.5--40 Hz filtering;
- 5--30 minute preictal interval;
- 10-second preictal sampling cadence;
- 5-minute interictal sampling cadence;
- four-hour postictal exclusion;
- within-window channel-wise z-normalization;
- windows with more than two near-flat channels removed.

The preprocessing output is one feature and one label array per case:

```text
chb01_X.npy   shape: (windows, 18, 5120), float32
chb01_y.npy   shape: (windows,), binary integer labels
```

Configure folders with command-line arguments to
`preprocess_chbmit.py`. The equivalent environment variables are
`CHBMIT_RAW_DIR`, `CHBMIT_PREPROCESSED_DIR`, and `CHBMIT_TEMP_DIR`.

## Siena

Source: [Siena Scalp EEG Database v1.0.0](https://physionet.org/content/siena-scalp-eeg/1.0.0/), DOI [10.13026/5d4a-j060](https://doi.org/10.13026/5d4a-j060).

The Siena release uses referential labels and 512 Hz sampling. The preprocessing
script reconstructs the same 18 bipolar derivations, filters and resamples to
256 Hz, and applies the CHB-MIT window rules. Thirteen participants with both
preictal and interictal windows enter the external-transfer summary.

The relevant environment variables are `SIENA_RAW_DIR`,
`SIENA_PREPROCESSED_DIR`, and `SIENA_TEMP_DIR`; all can also be supplied as
command-line paths.

## FPD_300 interpretation

Interictal windows are sampled every 300 seconds. Therefore the reported
`FPD_300/h` value is:

```text
false-positive window decisions / (number of interictal windows * 300 / 3600)
```

This cadence adjustment is useful for comparing window-level operating points,
but it does not reconstruct continuous monitoring time and must not be called a
clinical false-alarm rate.

## Curated repository artifacts

`results/strict_subject_predictions/` contains only held-out labels, scores,
and public case IDs. `results/manuscript_source_data/` contains exact plotting
tables. Raw EEG, preprocessed signal arrays, feature caches, and checkpoints are
excluded by `.gitignore`.
