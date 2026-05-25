# Data notes

The repository does not ship the EEG data or preprocessed arrays.

## CHB-MIT

Expected raw-data folder:

```text
D:\chbmit_data
```

Expected preprocessed folder:

```text
D:\chbmit_preprocessed
```

The main preprocessing rule uses 20 s windows, a 5-30 min preictal horizon, 10 s preictal stride, 5 min interictal stride, 0.5-40 Hz filtering, and 18 bipolar channels.

## Siena

Expected raw-data folder:

```text
D:\Siena\siena-scalp-eeg-1.0.0
```

Expected preprocessed folder:

```text
D:\siena_preprocessed
```

Siena uses referential EDF labels and 512 Hz sampling. The preprocessing script reconstructs the CHB-MIT-style bipolar derivations and resamples to 256 Hz before window extraction.

## Results

Scripts write to:

```text
D:\seizure_results
```

Keep result files out of the repository unless you add a small, curated table for documentation.
