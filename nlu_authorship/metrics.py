"""Binary discrimination and calibration metrics with explicit fixed policies."""
from __future__ import annotations

import numpy as np


def checked_labels(labels):
    values = np.asarray(labels)
    if values.ndim != 1 or not len(values) or not np.isin(values, (0, 1)).all():
        raise ValueError("Provide a nonempty vector of binary labels")
    return values.astype(np.int64)


def checked_probabilities(probabilities, count=None):
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim == 2 and values.shape[1] == 1:
        values = values[:, 0]
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Probabilities must be a finite vector in [0,1]")
    if count is not None and len(values) != count:
        raise ValueError("Labels and predictions have different counts")
    return values


def sigmoid(margins):
    margins = np.asarray(margins, dtype=np.float64)
    if margins.ndim != 1 or not len(margins) or not np.isfinite(margins).all():
        raise ValueError("Logit margins must be a finite nonempty vector")
    out = np.empty_like(margins)
    positive = margins >= 0
    out[positive] = 1 / (1 + np.exp(-margins[positive]))
    weights = np.exp(margins[~positive])
    out[~positive] = weights / (1 + weights)
    return np.clip(out, 1e-15, 1 - 1e-15)


def logit_margins(logits):
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 2 or logits.shape[1] != 2 or not len(logits) or not np.isfinite(logits).all():
        raise ValueError("Expected finite [N,2] binary logits")
    return logits[:, 1] - logits[:, 0]


def probability_logits(probabilities):
    probabilities = np.clip(checked_probabilities(probabilities), 1e-15, 1 - 1e-15)
    return np.stack([np.log1p(-probabilities), np.log(probabilities)], axis=1)


def auc_roc(labels, probabilities):
    labels = checked_labels(labels)
    p = checked_probabilities(probabilities, len(labels))
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if not positives or not negatives:
        return None
    order = np.argsort(p, kind="stable")
    ranks = np.empty(len(p), dtype=np.float64)
    start = 0
    while start < len(p):
        end = start + 1
        while end < len(p) and p[order[end]] == p[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2
        start = end
    return float((ranks[labels == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def f1_from_confusion(tn, fp, fn, tp):
    f1_0 = 2 * tn / (2 * tn + fp + fn) if 2 * tn + fp + fn else 0.0
    f1_1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return (f1_0 + f1_1) / 2, f1_0, f1_1


def binary_metrics(labels, probabilities, threshold=.5):
    y = checked_labels(labels)
    p = checked_probabilities(probabilities, len(y))
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Decision threshold must be in [0,1]")
    prediction = (p >= threshold).astype(np.int64)
    tn = int(np.sum((y == 0) & (prediction == 0)))
    fp = int(np.sum((y == 0) & (prediction == 1)))
    fn = int(np.sum((y == 1) & (prediction == 0)))
    tp = int(np.sum((y == 1) & (prediction == 1)))
    macro, f1_0, f1_1 = f1_from_confusion(tn, fp, fn, tp)
    clipped = np.clip(p, 1e-15, 1 - 1e-15)
    bins = np.minimum((p * 10).astype(int), 9)
    details = []
    ece = 0.0
    for index in range(10):
        mask = bins == index
        count = int(mask.sum())
        mean_probability = float(p[mask].mean()) if count else None
        positive_fraction = float(y[mask].mean()) if count else None
        if count:
            ece += count / len(y) * abs(mean_probability - positive_fraction)
        details.append({"bin": index, "lower": index / 10, "upper": (index + 1) / 10,
                        "upper_inclusive": index == 9, "count": count,
                        "mean_positive_probability": mean_probability, "positive_fraction": positive_fraction})
    return {"count": len(y), "correct": tn + tp, "accuracy": (tn + tp) / len(y),
            "macro_f1": macro, "class_f1": {"0": f1_0, "1": f1_1}, "same_author_f1": f1_1,
            "confusion": [[tn, fp], [fn, tp]], "class_support": {"0": tn + fp, "1": fn + tp},
            "class_recall": {"0": tn / (tn + fp) if tn + fp else None,
                             "1": tp / (fn + tp) if fn + tp else None},
            "roc_auc": auc_roc(y, p), "log_loss": float(-(y * np.log(clipped) + (1 - y) * np.log1p(-clipped)).mean()),
            "brier_score": float(np.mean((p - y) ** 2)), "ece10": ece,
            "ece10_policy": "Ten fixed equal-width bins of positive-class probability; weighted absolute mean probability minus positive-label frequency; empty bins contribute zero.",
            "ece10_bins": details, "threshold": float(threshold), "decision_rule": "positive when probability >= threshold",
            "macro_f1_policy": "Mean F1 of both binary classes; undefined class F1=0",
            "roc_auc_policy": "Average ranks for tied scores; null if one class is absent"}


def confidence_accuracy_bins(labels, probabilities, threshold=.5, bins=10):
    """Observed accuracy by confidence of the threshold-selected class.

    This table differs from ECE's positive-probability/label-frequency bins.
    A threshold different from0.5 can select a class with confidence below0.5.
    """
    y = checked_labels(labels)
    p = checked_probabilities(probabilities, len(y))
    if not np.isfinite(threshold) or not 0 <= threshold <= 1 or not isinstance(bins, int) or bins < 1:
        raise ValueError("Invalid confidence-bin count or decision threshold")
    prediction = (p >= threshold).astype(np.int64)
    confidence = np.where(prediction == 1, p, 1 - p)
    bin_ids = np.minimum((confidence * bins).astype(int), bins - 1)
    result = []
    for index in range(bins):
        mask = bin_ids == index
        count = int(mask.sum())
        correct = int(np.sum(prediction[mask] == y[mask]))
        result.append({"bin": index, "lower": index / bins, "upper": (index + 1) / bins,
                       "upper_inclusive": index == bins - 1, "count": count, "correct": correct,
                       "accuracy": correct / count if count else None,
                       "mean_confidence": float(confidence[mask].mean()) if count else None})
    return result


def swapped_diagnostics(logits_ab, logits_ba, *, temperature=1.0, threshold=.5):
    ab, ba = logit_margins(logits_ab), logit_margins(logits_ba)
    if len(ab) != len(ba) or not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Orientation counts or temperature are invalid")
    p_ab, p_ba = sigmoid(ab / temperature), sigmoid(ba / temperature)
    gap = np.abs(p_ab - p_ba)
    mismatch = (p_ab >= threshold) != (p_ba >= threshold)
    return {"count": len(ab), "mean_absolute_probability_gap": float(gap.mean()),
            "p95_absolute_probability_gap": float(np.percentile(gap, 95)),
            "max_absolute_probability_gap": float(gap.max()),
            "prediction_mismatch_count": int(mismatch.sum()), "prediction_mismatch_fraction": float(mismatch.mean()),
            "temperature": float(temperature), "threshold": float(threshold)}
