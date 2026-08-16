# Curated reproducibility artifacts

This directory contains small derived artifacts, not raw EEG.

- `manuscript_source_data/` contains the exact CSV tables supplied with the
  manuscript and read by its LaTeX figures and numerical summaries.
- `strict_subject_predictions/` contains 100 compressed held-out prediction
  archives (five models x 20 fixed seeds). Each archive has only three arrays:
  binary labels (`y_true`), model scores (`y_prob`), and public CHB-MIT case
  identifiers (`patient_ids`). No signal samples, clinical records, or model
  checkpoints are included.

The prediction archives were exported from the strict 22-subject-group runs.
`chb01` and `chb21`, which correspond to one individual, are bound to the same
partition for every seed. Run the following command from the repository root to
validate all archives and regenerate the low-FPD source table:

```bash
python run.py scripts/analysis/work_J_far_constrained_sensitivity.py \
  --source-data-out outputs/strict_low_far_per_seed.csv
```

The checked-in tables remain the publication record. Files under `outputs/`
are ignored so that local verification does not modify the record.
