"""FAR-constrained sensitivity operating-point analysis.

For each saved PI seed/model prediction vector, choose the most sensitive
test-set threshold whose false-alarm rate is at or below a clinical ceiling
(default 0.2/h). This is a descriptive ROC/FAR operating-point summary, not a
deployable validation-threshold rule.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from data_utils import SEEDS
from eval_utils import false_alarm_rate, full_evaluate
from work_E_permutation_null import (
    DEEP_MODELS,
    get_deep_seed_predictions,
    get_psd_seed_predictions,
)


RESULTS_ROOT = Path(r"D:\seizure_results")
OUT_DIR = RESULTS_ROOT / "analysis_outputs" / "work_J_far_constrained_sensitivity"

MODEL_ORDER = ["psd_lda", "1dcnn", "eegnet", "tcn", "eeg_conformer"]
MODEL_LABELS = {
    "psd_lda": "PSD+LDA",
    "1dcnn": "1D-CNN",
    "eegnet": "EEGNet",
    "tcn": "TCN",
    "eeg_conformer": "EEG-Conformer",
}


def threshold_for_far(y_true: np.ndarray, y_prob: np.ndarray, far_ceiling: float) -> tuple[float, dict]:
    """Return the threshold with maximum sensitivity subject to FAR <= ceiling."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    finite_prob = y_prob[np.isfinite(y_prob)]
    if len(finite_prob) == 0:
        threshold = float("inf")
        return threshold, full_evaluate(y_true, y_prob, threshold, stride_s=300)

    # At threshold > max probability, no window is flagged. Include this so a
    # feasible threshold always exists even for pathological score vectors.
    candidates = np.unique(finite_prob)
    candidates = np.concatenate(([float(finite_prob.max() + 1e-6)], candidates[::-1]))

    best_threshold = float(candidates[0])
    best_metrics = full_evaluate(y_true, y_prob, best_threshold, stride_s=300)

    for threshold in candidates:
        far = false_alarm_rate(y_true, y_prob, float(threshold), stride_s=300)
        if far <= far_ceiling + 1e-12:
            metrics = full_evaluate(y_true, y_prob, float(threshold), stride_s=300)
            if (
                metrics["sensitivity"] > best_metrics["sensitivity"]
                or (
                    np.isclose(metrics["sensitivity"], best_metrics["sensitivity"])
                    and metrics["far"] > best_metrics["far"]
                )
            ):
                best_threshold = float(threshold)
                best_metrics = metrics

    return best_threshold, best_metrics


def predictions_for(model_key: str, seed: int, force_predictions: bool = False):
    if model_key == "psd_lda":
        return get_psd_seed_predictions(seed, force=force_predictions)
    if model_key not in DEEP_MODELS:
        raise ValueError(f"Unknown model key: {model_key}")
    return get_deep_seed_predictions(model_key, seed, force=force_predictions)


def summarize(values: pd.Series) -> dict:
    values = pd.to_numeric(values, errors="coerce")
    return {
        "mean": float(values.mean()),
        "sd": float(values.std(ddof=1)),
        "median": float(values.median()),
        "q25": float(values.quantile(0.25)),
        "q75": float(values.quantile(0.75)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def run(far_ceiling: float, force_predictions: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for model_key in MODEL_ORDER:
        for seed in SEEDS:
            y_true, y_prob, patient_ids = predictions_for(model_key, seed, force_predictions)
            threshold, metrics = threshold_for_far(y_true, y_prob, far_ceiling)
            rows.append(
                {
                    "model_key": model_key,
                    "model": MODEL_LABELS[model_key],
                    "seed": seed,
                    "far_ceiling": far_ceiling,
                    "threshold": threshold,
                    "auc": metrics["auc"],
                    "sensitivity": metrics["sensitivity"],
                    "specificity": metrics["specificity"],
                    "precision": metrics["precision"],
                    "f1": metrics["f1"],
                    "far": metrics["far"],
                    "event_sensitivity": metrics["event_sensitivity"],
                    "n_events": metrics["n_events"],
                    "n_windows": int(len(y_true)),
                    "n_preictal": int((np.asarray(y_true) == 1).sum()),
                    "n_interictal": int((np.asarray(y_true) == 0).sum()),
                    "n_patients": int(len(np.unique(patient_ids.astype(str)))),
                }
            )

    per_seed = pd.DataFrame(rows)
    summary_rows = []
    for (model_key, model), sub in per_seed.groupby(["model_key", "model"], sort=False):
        row = {
            "model_key": model_key,
            "model": model,
            "n_seeds": int(len(sub)),
            "far_ceiling": far_ceiling,
        }
        for metric in ["auc", "sensitivity", "specificity", "precision", "f1", "far", "event_sensitivity", "threshold"]:
            stats = summarize(sub[metric])
            for stat_name, value in stats.items():
                row[f"{metric}_{stat_name}"] = value
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    return per_seed, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--far-ceiling", type=float, default=0.2)
    parser.add_argument("--force-predictions", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    per_seed, summary = run(args.far_ceiling, args.force_predictions)

    per_seed.to_csv(args.out_dir / "far_constrained_per_seed.csv", index=False)
    summary.to_csv(args.out_dir / "far_constrained_summary.csv", index=False)

    report = {
        "far_ceiling": args.far_ceiling,
        "n_rows": int(len(per_seed)),
        "note": (
            "Thresholds are selected post hoc on each held-out test score vector "
            "to maximize window-level sensitivity subject to FAR <= ceiling."
        ),
        "summary": summary.to_dict(orient="records"),
    }
    (args.out_dir / "far_constrained_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Saved outputs to {args.out_dir}")
    print(
        summary[
            [
                "model",
                "sensitivity_mean",
                "sensitivity_sd",
                "far_mean",
                "far_sd",
                "event_sensitivity_mean",
                "event_sensitivity_sd",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
