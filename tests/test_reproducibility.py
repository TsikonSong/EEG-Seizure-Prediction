"""Fast audits for the checked-in strict subject-grouped artifacts."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))

from splits import SEEDS, VALID_PATIENTS, make_subject_splits  # noqa: E402
from work_J_far_constrained_sensitivity import (  # noqa: E402
    manuscript_source_data,
    run,
)


SOURCE_DATA = ROOT / "results" / "manuscript_source_data"
PREDICTIONS = ROOT / "results" / "strict_subject_predictions"


class StrictSubjectSplitTests(unittest.TestCase):
    def test_all_fixed_seeds_have_no_subject_overlap(self) -> None:
        for seed in SEEDS:
            train, validation, test = make_subject_splits(seed)
            self.assertEqual(set(train) | set(validation) | set(test), set(VALID_PATIENTS))
            self.assertFalse(set(train) & set(validation))
            self.assertFalse(set(train) & set(test))
            self.assertFalse(set(validation) & set(test))
            partition = {
                case: name
                for name, cases in (
                    ("train", train),
                    ("validation", validation),
                    ("test", test),
                )
                for case in cases
            }
            self.assertEqual(partition["chb01"], partition["chb21"])

    def test_checked_partition_manifest_matches_split_function(self) -> None:
        manifest = pd.read_csv(SOURCE_DATA / "strict_partition_manifest.csv")
        self.assertEqual(set(manifest["seed"]), set(SEEDS))
        for row in manifest.itertuples(index=False):
            expected = make_subject_splits(int(row.seed))
            observed = (
                str(row.train_patients).split(","),
                str(row.val_patients).split(","),
                str(row.test_patients).split(","),
            )
            self.assertEqual(observed, expected)


class CuratedPredictionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.per_seed, _ = run(PREDICTIONS, 0.2)

    def test_all_archives_reproduce_low_fpd_source_data(self) -> None:
        observed = manuscript_source_data(self.per_seed)
        expected = pd.read_csv(SOURCE_DATA / "strict_low_far_per_seed.csv")
        assert_frame_equal(observed, expected, check_exact=False, atol=1e-15, rtol=0)

    def test_archive_auc_matches_tier3_source_data(self) -> None:
        observed = self.per_seed[["model", "seed", "auc"]].copy()
        expected = pd.read_csv(SOURCE_DATA / "tier3_per_seed.csv")[
            ["model", "seed", "auc"]
        ]
        merged = observed.merge(
            expected,
            on=["model", "seed"],
            suffixes=("_archive", "_source"),
            validate="one_to_one",
        )
        self.assertEqual(len(merged), 100)
        self.assertLessEqual(
            float(np.max(np.abs(merged["auc_archive"] - merged["auc_source"]))),
            1e-12,
        )

    def test_siena_internal_auc_matches_strict_psd_auc(self) -> None:
        siena = pd.read_csv(SOURCE_DATA / "siena_strict_psd.csv")
        tier3 = pd.read_csv(SOURCE_DATA / "tier3_per_seed.csv")
        psd = tier3.loc[tier3["model"] == "PSD+LDA", ["seed", "auc"]]
        merged = siena.merge(psd, on="seed", validate="one_to_one")
        self.assertEqual(len(merged), 20)
        # Siena reruns refit LDA from the cached float32 PSD features; the
        # independently exported strict prediction arrays can differ at about
        # the sixth decimal place across scikit-learn builds.
        self.assertLessEqual(
            float(np.max(np.abs(merged["chb_internal_auc"] - merged["auc"]))),
            2e-6,
        )


if __name__ == "__main__":
    unittest.main()
