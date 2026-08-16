# Windows runners

These scripts queue the longer Windows jobs used in the benchmark.

By default they write beneath:

```text
D:\seizure_results
```

Set `SEIZURE_PYTHON` if you want a specific interpreter; otherwise the scripts
use `python` from the active shell. Set `SEIZURE_RESULTS_DIR` to override the
default result root. The historically named
`run_submission_blocker_experiments.ps1` now runs the leakage and low-FPD
sensitivity checks with the publication-facing subject-grouped prediction path.
