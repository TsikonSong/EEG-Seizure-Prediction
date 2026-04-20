import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, confusion_matrix, precision_score


# ---------------------------------------------------------------------------
# Threshold selection
# ---------------------------------------------------------------------------

def find_youden_threshold(y_true, y_prob):
    if roc_auc_score(y_true, y_prob) < 0.5:
        return 0.5  # classifier worse than random; fall back to neutral threshold
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j_scores = tpr - fpr   # Youden's J = sensitivity - (1 - specificity) = tpr - fpr
    best_idx = np.argmax(j_scores)
    return float(thresholds[best_idx])


# ---------------------------------------------------------------------------
# Main evaluation (works with raw probabilities, not models)
# ---------------------------------------------------------------------------

def evaluate_at_threshold(y_true, y_prob, threshold):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    auc = roc_auc_score(y_true, y_prob)
    f1  = f1_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        'auc':         float(auc),
        'sensitivity': float(sensitivity),
        'specificity': float(specificity),
        'precision':   float(prec),
        'f1':          float(f1),
        'threshold':   float(threshold),
    }


# ---------------------------------------------------------------------------
# False Alarm Rate
# ---------------------------------------------------------------------------

def false_alarm_rate(y_true, y_prob, threshold, stride_s=300):
    """False alarms per hour; stride_s=300 matches 5-min interictal sampling."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    mask_inter = (y_true == 0)
    n_inter = mask_inter.sum()
    n_false = ((y_pred == 1) & mask_inter).sum()

    if n_inter == 0:
        return float('nan')

    interictal_hours = (n_inter * stride_s) / 3600
    return float(n_false / interictal_hours)


# ---------------------------------------------------------------------------
# Event-level sensitivity
# ---------------------------------------------------------------------------

def event_level_sensitivity(y_true, y_prob, threshold):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    # Find contiguous runs of preictal windows (label=1)
    is_pre = (y_true == 1).astype(int)
    # Detect run boundaries: diff != 0 indicates transition
    diff = np.diff(is_pre, prepend=0, append=0)
    run_starts = np.where(diff == 1)[0]   # 0→1 transitions
    run_ends   = np.where(diff == -1)[0]  # 1→0 transitions

    n_events = len(run_starts)
    if n_events == 0:
        return 0.0, 0

    n_predicted = 0
    for s, e in zip(run_starts, run_ends):
        # Check if any window in this preictal run was predicted positive
        if y_pred[s:e].any():
            n_predicted += 1

    return float(n_predicted / n_events), n_events


# ---------------------------------------------------------------------------
# Per-patient evaluation
# ---------------------------------------------------------------------------

def per_patient_evaluate(y_true, y_prob, patient_ids, threshold, stride_s=300):
    y_true      = np.asarray(y_true)
    y_prob      = np.asarray(y_prob)
    patient_ids = np.asarray(patient_ids)

    per_pt = {}
    for pid in np.unique(patient_ids):
        mask   = patient_ids == pid
        yt, yp = y_true[mask], y_prob[mask]
        y_pred = (yp >= threshold).astype(int)

        # AUC — requires both classes present
        if len(np.unique(yt)) < 2:
            auc = float('nan')
        else:
            auc = float(roc_auc_score(yt, yp))

        # Sensitivity (window-level)
        tp  = int(((y_pred == 1) & (yt == 1)).sum())
        fn  = int(((y_pred == 0) & (yt == 1)).sum())
        sen = tp / (tp + fn) if (tp + fn) > 0 else float('nan')

        # FAR
        far = false_alarm_rate(yt, yp, threshold, stride_s)

        per_pt[pid] = {
            'auc':         auc,
            'sensitivity': float(sen),
            'far':         far,
        }

    def _stats(key):
        vals = [v[key] for v in per_pt.values() if not np.isnan(v[key])]
        if not vals:
            return {'mean': float('nan'), 'std': float('nan')}
        return {'mean': float(np.mean(vals)), 'std': float(np.std(vals))}

    summary = {
        'auc':         _stats('auc'),
        'sensitivity': _stats('sensitivity'),
        'far':         _stats('far'),
    }
    return per_pt, summary


# ---------------------------------------------------------------------------
# Combined evaluation (convenience wrapper)
# ---------------------------------------------------------------------------

def full_evaluate(y_true, y_prob, threshold, stride_s=300):
    metrics = evaluate_at_threshold(y_true, y_prob, threshold)
    metrics['far'] = false_alarm_rate(y_true, y_prob, threshold, stride_s)

    evt_sen, n_evt = event_level_sensitivity(y_true, y_prob, threshold)
    metrics['event_sensitivity'] = evt_sen
    metrics['n_events'] = n_evt

    return metrics
