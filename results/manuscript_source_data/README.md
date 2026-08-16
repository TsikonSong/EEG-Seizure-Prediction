# Figure data

These compact tables contain the exact observations plotted by the
LaTeX/PGFPlots figures. No points were simulated or reconstructed.

- tier3_per_seed.csv: 20 strict subject-grouped CHB-MIT partitions for each model.
- tier1_summary.csv: direct-overlap software stress-test results, including the EEGNet replication count.
- tier2_paired.csv: paired chronological and random within-case AUCs for ten cases.
- subject_balanced_per_group.csv: subject-group AUCs averaged over all strict test appearances, with each of 22 subject groups contributing equally.
- strict_low_far_per_seed.csv: post hoc low-decision operating points from the strict subject-grouped prediction arrays.
- strict_partition_manifest.csv: the exact training, validation and test case identifiers for all 20 strict seeds; chb01 and chb21 always move together.
- siena_strict_psd.csv: strict PSD+LDA CHB-MIT test AUCs and direct Siena transfer results, paired by seed.

The tables are deliberately small, auditable plotting inputs. The strict
low-decision table can be regenerated from the checked-in held-out prediction
archives with `work_J_far_constrained_sensitivity.py`. The Siena table can be
regenerated from the public CHB-MIT and Siena datasets with
`work_L_siena_external_psd_lda.py`. Neither path requires or modifies raw data
inside this repository.
