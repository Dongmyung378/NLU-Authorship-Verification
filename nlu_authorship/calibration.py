"""Fit temperature, decision threshold and abstention on calibration data only."""
from __future__ import annotations

import math
import numpy as np

from .metrics import (binary_metrics, checked_labels, checked_probabilities,
                      f1_from_confusion, logit_margins, sigmoid, swapped_diagnostics)


def selected_margins(logits_ab, logits_ba=None, method="ab"):
    ab = logit_margins(logits_ab)
    if method == "ab":
        return ab
    if method != "symmetric" or logits_ba is None:
        raise ValueError("Use ab or symmetric with both orientation logits")
    ba = logit_margins(logits_ba)
    if len(ab) != len(ba):
        raise ValueError("Orientation prediction counts differ")
    return (ab + ba) / 2


def _require_calibration(split):
    if split != "calibration":
        raise ValueError("Fit probability/threshold/abstention on calibration only")


def fit_temperature(margins, labels, *, split="calibration", minimum=.05, maximum=20.0):
    _require_calibration(split)
    margins = np.asarray(margins, np.float64)
    y = checked_labels(labels)
    if margins.shape != y.shape or not np.isfinite(margins).all() or set(y.tolist()) != {0, 1}:
        raise ValueError("Temperature fit requires finite margins and both label classes")
    if not 0 < minimum <= 1 <= maximum or minimum == maximum:
        raise ValueError("Temperature bounds must include identity and have positive width")
    def nll(log_temperature):
        z = margins / math.exp(log_temperature)
        return float(np.mean(np.logaddexp(0, z) - y * z))
    left, right = math.log(minimum), math.log(maximum)
    phi = (math.sqrt(5) - 1) / 2
    a, b = right - phi * (right - left), left + phi * (right - left)
    fa, fb = nll(a), nll(b)
    for _ in range(96):
        if fa <= fb:
            right, b, fb = b, a, fa
            a = right - phi * (right - left)
            fa = nll(a)
        else:
            left, a, fa = a, b, fb
            b = left + phi * (right - left)
            fb = nll(b)
    candidates = [0.0, math.log(minimum), math.log(maximum), (left + right) / 2]
    best = min(candidates, key=lambda value: (nll(value), abs(value)))
    identity_loss = nll(0)
    selected_loss = nll(best)
    temperature = math.exp(best) if selected_loss <= identity_loss else 1.0
    return {"temperature": temperature, "minimum": minimum, "maximum": maximum,
            "identity_calibration_nll": identity_loss, "fitted_calibration_nll": nll(math.log(temperature)),
            "fitting_split": split, "optimization": "Deterministic golden-section search in log-temperature with identity and bounds retained; 96 iterations."}


def fit_threshold(probabilities, labels, *, split="calibration"):
    _require_calibration(split)
    y = checked_labels(labels)
    p = checked_probabilities(probabilities, len(y))
    order = np.argsort(-p, kind="stable")
    positives, negatives = int(y.sum()), int((1 - y).sum())
    candidates = [(binary_metrics(y, p, 1.0)["macro_f1"], 1.0)]
    tp = fp = 0
    start = 0
    while start < len(y):
        end = start + 1
        while end < len(y) and p[order[end]] == p[order[start]]:
            end += 1
        labels_group = y[order[start:end]]
        tp += int(labels_group.sum())
        fp += len(labels_group) - int(labels_group.sum())
        candidates.append((f1_from_confusion(negatives - fp, fp, positives - tp, tp)[0], float(p[order[start]])))
        start = end
    candidates.append((binary_metrics(y, p, .5)["macro_f1"], .5))
    best_score, threshold = min(candidates, key=lambda pair: (-pair[0], abs(pair[1] - .5), pair[1]))
    return {"threshold": threshold, "calibration_macro_f1": best_score,
            "fitting_split": split, "objective": "Maximum both-class macro F1; ties prefer threshold closest to0.5, then lower threshold."}


def fit_abstention(probabilities, labels, *, threshold=.5, split="calibration", target_coverage=.8, target_accuracy=.9):
    _require_calibration(split)
    y = checked_labels(labels)
    p = checked_probabilities(probabilities, len(y))
    if not 0 < target_coverage <= 1 or not 0 <= target_accuracy <= 1 or not 0 <= threshold <= 1:
        raise ValueError("Invalid abstention target or threshold")
    prediction = (p >= threshold).astype(int)
    confidence = np.where(prediction == 1, p, 1 - p)
    order = np.argsort(-confidence, kind="stable")
    candidates, correct = [], 0
    start = 0
    while start < len(y):
        end = start + 1
        while end < len(y) and confidence[order[end]] == confidence[order[start]]:
            end += 1
        correct += int(np.sum(prediction[order[start:end]] == y[order[start:end]]))
        coverage = end / len(y)
        if coverage + 1e-15 >= target_coverage:
            candidates.append({"confidence_threshold": float(confidence[order[start]]), "accepted_count": end,
                               "coverage": coverage, "accepted_accuracy": correct / end})
        start = end
    met = [item for item in candidates if item["accepted_accuracy"] + 1e-15 >= target_accuracy]
    selected = max(met, key=lambda item: (item["coverage"], item["accepted_accuracy"])) if met else max(candidates, key=lambda item: (item["accepted_accuracy"], item["coverage"]))
    return {**selected, "target_coverage": target_coverage, "target_accepted_accuracy": target_accuracy,
            "target_achieved_on_calibration": bool(met), "fitting_split": split,
            "confidence_policy": "Probability assigned to threshold-selected label; accept when confidence>=frozen cutoff.",
            "objective": "Maximum coverage meeting both targets; if infeasible, best observed accuracy with required coverage, then maximum coverage. Calibration fit is not held-out evidence."}


def fit_calibration(logits_ab, labels, logits_ba=None, method="ab", *, split="calibration", ids=None,
                    minimum_temperature=.05, maximum_temperature=20.0, target_coverage=.8, target_accuracy=.9):
    _require_calibration(split)
    margins = selected_margins(logits_ab, logits_ba, method)
    y = checked_labels(labels)
    if len(margins) != len(y):
        raise ValueError("Calibration prediction count differs from labels")
    if ids is not None and (len(ids) != len(y) or len(set(ids)) != len(ids)):
        raise ValueError("Calibration IDs must uniquely match every record")
    temp = fit_temperature(margins, y, split=split, minimum=minimum_temperature, maximum=maximum_temperature)
    p = sigmoid(margins / temp["temperature"])
    threshold = fit_threshold(p, y, split=split)
    abstention = fit_abstention(p, y, threshold=threshold["threshold"], split=split,
                                target_coverage=target_coverage, target_accuracy=target_accuracy)
    from .data import canonical_sha256
    return {"schema_version": 1, "fitting_split": split, "count": len(y), "inference_method": method,
            "temperature": temp["temperature"], "threshold": threshold["threshold"],
            "temperature_fit": temp, "threshold_fit": threshold, "abstention": abstention,
            "calibration_records_sha256": canonical_sha256([{"id": i, "label": int(label)} for i, label in zip(ids, y)]) if ids is not None else None,
            "raw_calibration_metrics": binary_metrics(y, sigmoid(margins)),
            "fitted_calibration_metrics": binary_metrics(y, p, threshold["threshold"]),
            "scope": "All metrics in this fit report describe calibration data; final evaluation is separate."}


def validate_calibration(calibration):
    if calibration.get("schema_version") != 1 or calibration.get("fitting_split") != "calibration":
        raise ValueError("Expected frozen calibration parameters fitted on calibration split")
    temperature = float(calibration["temperature"])
    if not np.isfinite(temperature) or temperature <= 0 or not 0 <= calibration["threshold"] <= 1:
        raise ValueError("Invalid frozen calibration parameters")
    abstention = calibration.get("abstention", {})
    cutoff = abstention.get("confidence_threshold")
    coverage, accuracy = abstention.get("target_coverage"), abstention.get("target_accepted_accuracy")
    if cutoff is None or not np.isfinite(cutoff) or not 0 <= cutoff <= 1 or coverage is None or not np.isfinite(coverage) or not 0 < coverage <= 1 or accuracy is None or not np.isfinite(accuracy) or not 0 <= accuracy <= 1:
        raise ValueError("Invalid frozen abstention policy")
    if calibration.get("inference_method") not in {"ab", "symmetric"}:
        raise ValueError("Invalid frozen input-order method")
    return calibration


def apply_calibration(logits_ab, calibration, logits_ba=None):
    validate_calibration(calibration)
    temperature = float(calibration["temperature"])
    return sigmoid(selected_margins(logits_ab, logits_ba, calibration["inference_method"]) / temperature)


def select_model_policy(predictions, *, split="selection", minimum_improvement=.002, baseline_name="ce", alternative_name="custom_asl"):
    if split != "selection" or not np.isfinite(minimum_improvement) or minimum_improvement < .002:
        raise ValueError("Model/method selection requires selection split and a>=0.002 declared improvement gate")
    if baseline_name not in predictions:
        raise ValueError("Matched CE reference is required")
    outcomes = {}
    reference_ids = predictions[baseline_name]["ids"]
    reference_labels = checked_labels(predictions[baseline_name]["labels"])
    for name, item in predictions.items():
        if item["ids"] != reference_ids or not np.array_equal(checked_labels(item["labels"]), reference_labels):
            raise ValueError("Selection comparisons must use the same record IDs and labels")
        ab = binary_metrics(reference_labels, sigmoid(selected_margins(item["logits_ab"])))
        ba = binary_metrics(reference_labels, sigmoid(selected_margins(item["logits_ba"])))
        symmetric = binary_metrics(reference_labels, sigmoid(selected_margins(item["logits_ab"], item["logits_ba"], "symmetric")))
        gain = symmetric["macro_f1"] - ab["macro_f1"]
        raw_order = swapped_diagnostics(item["logits_ab"], item["logits_ba"])
        max_logit_difference = float(np.max(np.abs(np.asarray(item["logits_ab"], np.float64) - np.asarray(item["logits_ba"], np.float64))))
        first = sigmoid(selected_margins(item["logits_ab"], item["logits_ba"], "symmetric"))
        second = sigmoid(selected_margins(item["logits_ba"], item["logits_ab"], "symmetric"))
        symmetric_gap = float(np.max(np.abs(first - second)))
        asymmetric = raw_order["prediction_mismatch_count"] > 0 or max_logit_difference > 1e-12
        improvement_gate = gain >= minimum_improvement
        invariance_no_harm_gate = asymmetric and symmetric_gap == 0 and gain >= -minimum_improvement
        method = "symmetric" if improvement_gate or invariance_no_harm_gate else "ab"
        outcomes[name] = {"ab": ab, "ba": ba, "symmetric": symmetric, "symmetric_macro_f1_gain": gain,
                          "raw_order_diagnostics": raw_order, "raw_orientation_max_logit_difference": max_logit_difference,
                          "symmetric_max_probability_gap": symmetric_gap,
                          "symmetric_improvement_gate_passed": improvement_gate,
                          "symmetric_invariance_no_harm_gate_passed": invariance_no_harm_gate,
                          "selected_method": method, "selected_macro_f1": (symmetric if method == "symmetric" else ab)["macro_f1"]}
    selected_name = baseline_name
    alternative_gain = None
    if alternative_name in outcomes:
        alternative_gain = outcomes[alternative_name]["selected_macro_f1"] - outcomes[baseline_name]["selected_macro_f1"]
        if alternative_gain >= minimum_improvement:
            selected_name = alternative_name
    return {"schema_version": 1, "selection_split": split, "metric": "macro_f1", "decision_threshold": .5,
            "minimum_improvement": minimum_improvement, "outcomes": outcomes,
            "alternative_macro_f1_gain": alternative_gain, "selected_model": selected_name,
            "alternative_macro_f1_gain_raw_ab": outcomes[alternative_name]["ab"]["macro_f1"] - outcomes[baseline_name]["ab"]["macro_f1"] if alternative_name in outcomes else None,
            "loss_comparison_scope": "Raw AB CE versus custom ASL uses the same orientation and0.5 threshold; alternative_macro_f1_gain separately compares validation-selected inference policies.",
            "symmetric_policy": "Choose symmetric if macro F1 gains>=0.002, or exact input-order invariance improves with macro F1 loss<=0.002; logit/probability/order diagnostics disclosed.",
            "selected_method": outcomes[selected_name]["selected_method"], "final_outcomes_used": False}
