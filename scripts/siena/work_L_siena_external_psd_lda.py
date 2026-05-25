"""PSD+LDA external-transfer probe from CHB-MIT to Siena.

This script runs the external-transfer check with a deliberately simple model.
It keeps the CHB-MIT PI training protocol and evaluates the trained PSD+LDA
classifier on the independently preprocessed Siena windows. No deep models are
trained.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import welch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, precision_score, roc_auc_score

from data_utils import DATA_DIR as CHB_DIR
from data_utils import SEEDS, make_patient_splits
from eval_utils import false_alarm_rate, find_youden_threshold


RESULTS_ROOT = Path(r"D:\seizure_results\siena_pilot")
OUT_DIR = RESULTS_ROOT / "psd_lda_external_transfer"
FEATURE_CACHE = OUT_DIR / "feature_cache"
SIENA_DIR = Path(r"D:\siena_preprocessed")
FS = 256
INTERICTAL_STRIDE_S = 300
FAR_TARGET = 0.2
BANDS = [(0.5, 4), (4, 8), (8, 13), (13, 30), (30, 40)]


def extract_psd(patient: str, data_dir: Path | str, cache_prefix: str, chunk_size: int = 256):
    FEATURE_CACHE.mkdir(parents=True, exist_ok=True)
    cache_path = FEATURE_CACHE / f"{cache_prefix}_{patient}_psd.npz"
    if cache_path.exists():
        data = np.load(cache_path, allow_pickle=False)
        return data["X"], data["y"]

    data_dir = Path(data_dir)
    x_raw = np.load(data_dir / f"{patient}_X.npy", mmap_mode="r")
    y = np.load(data_dir / f"{patient}_y.npy").astype(np.int64)
    chunks = []
    for start in range(0, len(y), chunk_size):
        chunk = np.asarray(x_raw[start:start + chunk_size], dtype=np.float32)
        freqs, pxx = welch(chunk, fs=FS, axis=-1, nperseg=512)
        features = [
            pxx[:, :, (freqs >= lo) & (freqs <= hi)].mean(axis=-1)
            for lo, hi in BANDS
        ]
        chunks.append(np.concatenate(features, axis=1).astype(np.float32))
    X = np.vstack(chunks)
    np.savez_compressed(cache_path, X=X, y=y)
    return X, y


def load_patients(patients, data_dir, cache_prefix):
    xs, ys, pids = [], [], []
    for patient in patients:
        X, y = extract_psd(patient, data_dir, cache_prefix)
        xs.append(X)
        ys.append(y)
        pids.extend([patient] * len(y))
    return np.vstack(xs), np.concatenate(ys), np.asarray(pids)


def siena_patients_with_both_classes():
    patients = []
    excluded = {}
    for y_path in sorted(SIENA_DIR.glob("*_y.npy")):
        patient = y_path.name.replace("_y.npy", "")
        y = np.load(y_path)
        counts = {"preictal": int((y == 1).sum()), "interictal": int((y == 0).sum())}
        if counts["preictal"] > 0 and counts["interictal"] > 0:
            patients.append(patient)
        else:
            excluded[patient] = counts
    return patients, excluded


def safe_auc(y_true, y_prob):
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def metrics_at_threshold(y_true, y_prob, threshold):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    precision = precision_score(y_true, y_pred, zero_division=0)
    return {
        "auc": safe_auc(y_true, y_prob),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision),
        "far": false_alarm_rate(y_true, y_prob, threshold, stride_s=INTERICTAL_STRIDE_S),
        "threshold": float(threshold),
    }


def threshold_for_far(y_true, y_prob, max_far=FAR_TARGET):
    candidates = np.unique(y_prob)
    candidates = np.r_[np.inf, candidates]
    best = None
    for threshold in candidates:
        m = metrics_at_threshold(y_true, y_prob, threshold)
        if np.isnan(m["far"]) or m["far"] > max_far:
            continue
        key = (m["sensitivity"], -m["far"], -float(threshold))
        if best is None or key > best[0]:
            best = (key, threshold, m)
    if best is None:
        return float("inf")
    return float(best[1])


def macro_patient_auc(y_true, y_prob, patient_ids):
    rows = []
    for patient in sorted(np.unique(patient_ids)):
        mask = patient_ids == patient
        rows.append({
            "patient": patient,
            "auc": safe_auc(y_true[mask], y_prob[mask]),
            "n": int(mask.sum()),
            "preictal": int((y_true[mask] == 1).sum()),
            "interictal": int((y_true[mask] == 0).sum()),
        })
    vals = [row["auc"] for row in rows if not np.isnan(row["auc"])]
    return rows, {
        "mean": float(np.mean(vals)) if vals else float("nan"),
        "std": float(np.std(vals)) if vals else float("nan"),
        "n_patients": len(vals),
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FEATURE_CACHE.mkdir(parents=True, exist_ok=True)

    siena_patients, siena_excluded = siena_patients_with_both_classes()
    X_siena, y_siena, pids_siena = load_patients(siena_patients, SIENA_DIR, "siena")

    seed_rows = []
    per_patient_by_seed = {}
    for seed in SEEDS:
        train_pts, val_pts, test_pts = make_patient_splits(seed)
        X_train, y_train, _ = load_patients(train_pts, CHB_DIR, "chb")
        X_val, y_val, _ = load_patients(val_pts, CHB_DIR, "chb")
        X_test, y_test, _ = load_patients(test_pts, CHB_DIR, "chb")

        clf = LinearDiscriminantAnalysis(solver="svd", priors=[0.5, 0.5])
        clf.fit(X_train, y_train)

        val_prob = clf.predict_proba(X_val)[:, 1]
        test_prob = clf.predict_proba(X_test)[:, 1]
        siena_prob = clf.predict_proba(X_siena)[:, 1]

        youden_threshold = find_youden_threshold(y_val, val_prob)
        far_threshold = threshold_for_far(y_val, val_prob, FAR_TARGET)
        chb_test_youden = metrics_at_threshold(y_test, test_prob, youden_threshold)
        siena_youden = metrics_at_threshold(y_siena, siena_prob, youden_threshold)
        siena_far = metrics_at_threshold(y_siena, siena_prob, far_threshold)
        per_patient, macro_auc = macro_patient_auc(y_siena, siena_prob, pids_siena)
        per_patient_by_seed[str(seed)] = per_patient

        seed_rows.append({
            "seed": seed,
            "chb_val_auc": safe_auc(y_val, val_prob),
            "chb_test_auc": chb_test_youden["auc"],
            "chb_test_youden_sensitivity": chb_test_youden["sensitivity"],
            "chb_test_youden_far": chb_test_youden["far"],
            "siena_pooled_auc": siena_youden["auc"],
            "siena_macro_patient_auc_mean": macro_auc["mean"],
            "siena_macro_patient_auc_std": macro_auc["std"],
            "siena_youden_threshold": youden_threshold,
            "siena_youden_sensitivity": siena_youden["sensitivity"],
            "siena_youden_specificity": siena_youden["specificity"],
            "siena_youden_precision": siena_youden["precision"],
            "siena_youden_far": siena_youden["far"],
            "chb_val_far02_threshold": far_threshold,
            "siena_far02_sensitivity": siena_far["sensitivity"],
            "siena_far02_specificity": siena_far["specificity"],
            "siena_far02_precision": siena_far["precision"],
            "siena_far02_far": siena_far["far"],
            "train_patients": ",".join(train_pts),
            "val_patients": ",".join(val_pts),
            "test_patients": ",".join(test_pts),
        })
        print(
            f"seed={seed} CHB_test_AUC={chb_test_youden['auc']:.3f} "
            f"Siena_AUC={siena_youden['auc']:.3f} "
            f"Siena_FAR0.2_sen={siena_far['sensitivity']:.3f} "
            f"Siena_FAR={siena_far['far']:.3f}"
        )

    df = pd.DataFrame(seed_rows)
    csv_path = OUT_DIR / "seed_results.csv"
    df.to_csv(csv_path, index=False)

    numeric = df.select_dtypes(include=[np.number])
    summary = {
        "analysis": "PSD+LDA CHB-MIT to Siena external-transfer probe",
        "chb_data_dir": str(CHB_DIR),
        "siena_data_dir": str(SIENA_DIR),
        "siena_included_patients": siena_patients,
        "siena_excluded_patients": siena_excluded,
        "n_siena_windows": int(len(y_siena)),
        "n_siena_preictal": int((y_siena == 1).sum()),
        "n_siena_interictal": int((y_siena == 0).sum()),
        "n_seeds": len(df),
        "mean": numeric.mean(numeric_only=True).to_dict(),
        "std": numeric.std(numeric_only=True, ddof=1).to_dict(),
        "per_patient_auc_by_seed": per_patient_by_seed,
    }
    summary_path = OUT_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {csv_path}")
    print(f"Wrote {summary_path}")
    print(json.dumps({
        "siena_pooled_auc_mean": summary["mean"]["siena_pooled_auc"],
        "siena_pooled_auc_std": summary["std"]["siena_pooled_auc"],
        "siena_far02_sensitivity_mean": summary["mean"]["siena_far02_sensitivity"],
        "siena_far02_far_mean": summary["mean"]["siena_far02_far"],
    }, indent=2))


if __name__ == "__main__":
    main()
