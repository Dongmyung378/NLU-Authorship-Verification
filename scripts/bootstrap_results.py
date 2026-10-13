"""Report predeclared paired document-component uncertainty from frozen scores.

This helper never loads a model or refits a decision. Its percentile intervals
condition on this corpus, its observed document components and one trained seed.
They do not quantify training-seed, new-author or new-domain uncertainty.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from nlu_authorship.data import canonical_sha256, file_sha256, load_manifest
from nlu_authorship.evaluation import source_inventory, validate_freeze


MODEL_NAMES = ("ce", "custom_asl", "stylometric")
BOOTSTRAP_PLAN = {
    "schema_version": 1,
    "method": "paired_document_component_percentile_bootstrap",
    "draws": 1000,
    "seed": 378,
    "confidence_level": 0.95,
    "resample_unit": "group_id",
    "group_draw_count": "number_of_locked_holdout_groups",
    "replacement": True,
    "paired_models": list(MODEL_NAMES),
    "metrics": ["accuracy", "macro_f1"],
    "paired_difference": "custom_asl_minus_ce_calibrated_macro_f1",
    "decision_role": "report_only_no_reselection",
}


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key: " + key)
        result[key] = value
    return result


def _invalid_number(value):
    raise ValueError("Nonfinite JSON number: " + value)


def _parse_json(text):
    return json.loads(text, object_pairs_hook=_unique_object,
                      parse_constant=_invalid_number)


def _read_json(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("A regular JSON evidence file is required: " + str(path))
    return _parse_json(path.read_text(encoding="utf-8"))


def group_draw_weights(group_ids, *, draws=1000, seed=378):
    """One shared multiplicity per whole group, not independent row resampling."""
    if not group_ids or draws < 1 or not all(isinstance(x, str) and x for x in group_ids):
        raise ValueError("Nonempty document group identities and positive draws are required")
    groups = sorted(set(group_ids))
    indices = {name: index for index, name in enumerate(groups)}
    row_groups = np.array([indices[name] for name in group_ids], dtype=np.int64)
    rng = np.random.Generator(np.random.PCG64(seed))
    weights = np.empty((draws, len(groups)), dtype=np.int64)
    for index in range(draws):
        sampled = rng.integers(0, len(groups), size=len(groups))
        weights[index] = np.bincount(sampled, minlength=len(groups))
    return groups, row_groups, weights


def _confusion_metrics(confusion):
    counts = np.asarray(confusion, dtype=np.int64)
    tn, fp, fn, tp = np.moveaxis(counts, -1, 0)
    total = counts.sum(axis=-1)
    if np.any(total <= 0):
        raise ValueError("Bootstrap draw has no evaluated pairs")
    denominator_0, denominator_1 = 2 * tn + fp + fn, 2 * tp + fp + fn
    f0 = np.divide(2 * tn, denominator_0, out=np.zeros_like(tn, dtype=float), where=denominator_0 != 0)
    f1 = np.divide(2 * tp, denominator_1, out=np.zeros_like(tp, dtype=float), where=denominator_1 != 0)
    return {"accuracy": (tn + tp) / total, "macro_f1": (f0 + f1) / 2}


def paired_group_bootstrap(labels, predictions, group_ids):
    """Fast binary confusion aggregation with identical group draws for all models."""
    y = np.asarray(labels)
    if y.ndim != 1 or len(y) != len(group_ids) or not len(y) or not np.isin(y, (0, 1)).all():
        raise ValueError("Aligned nonempty binary labels and groups are required")
    if set(predictions) != set(MODEL_NAMES):
        raise ValueError("All three frozen model predictions are required")
    groups, row_groups, weights = group_draw_weights(group_ids)
    component_counts = np.zeros((len(groups), len(MODEL_NAMES), 4), dtype=np.int64)
    for index, name in enumerate(MODEL_NAMES):
        p = np.asarray(predictions[name])
        if p.shape != y.shape or not np.isin(p, (0, 1)).all():
            raise ValueError("Every model must cover the same binary-labeled pairs")
        combined = row_groups * 4 + y.astype(np.int64) * 2 + p.astype(np.int64)
        component_counts[:, index, :] = np.bincount(combined, minlength=len(groups) * 4).reshape(-1, 4)
    sampled = np.einsum("dg,gmc->dmc", weights, component_counts, optimize=True)
    sample_metrics = _confusion_metrics(sampled)
    points = _confusion_metrics(component_counts.sum(axis=0))
    models = {}
    for index, name in enumerate(MODEL_NAMES):
        models[name] = {}
        for metric in BOOTSTRAP_PLAN["metrics"]:
            lo, hi = np.percentile(sample_metrics[metric][:, index], [2.5, 97.5], method="linear")
            models[name][metric] = {"point": float(points[metric][index]), "lower": float(lo), "upper": float(hi)}
    delta = sample_metrics["macro_f1"][:, 1] - sample_metrics["macro_f1"][:, 0]
    lo, hi = np.percentile(delta, [2.5, 97.5], method="linear")
    sizes = sampled[:, 0, :].sum(axis=1)
    return {
        "models": models,
        "paired_difference": {"direction": "custom_asl_minus_ce", "metric": "calibrated_macro_f1",
                              "point": float(points["macro_f1"][1] - points["macro_f1"][0]),
                              "lower": float(lo), "upper": float(hi)},
        "document_component_count": len(groups),
        "evaluated_pair_count": len(y),
        "sampled_pair_count": {"minimum": int(sizes.min()), "maximum": int(sizes.max()),
                               "median": float(np.median(sizes)), "variable_counts_allowed": True},
        "group_draw_weights_sha256": hashlib.sha256(weights.astype("<i8", copy=False).tobytes()).hexdigest(),
        "ordered_group_ids_sha256": canonical_sha256(groups),
        "shared_draws_for_all_models": True,
        "pair_weighting": "All pairs in each sampled group repeat by that group's multiplicity; metrics weight pairs equally.",
        "undefined_class_f1": 0.0,
    }


def _close(left, right, message):
    if isinstance(right, bool) or not isinstance(right, (float, int)) or not math.isfinite(right) or not math.isclose(left, right, abs_tol=1e-12, rel_tol=0):
        raise ValueError(message)


def bootstrap_results(manifest_path, freeze_path, summary_path, prediction_root, output_path):
    """Validate exact final evidence, then report uncertainty without decisions."""
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError("Bootstrap evidence is immutable; choose a new output path")
    manifest = load_manifest(manifest_path)
    frozen = validate_freeze(freeze_path, manifest)
    summary = _read_json(summary_path)
    if frozen.get("final_evaluation_split") != "locked_holdout":
        raise ValueError("Only frozen locked_holdout evaluation is eligible")
    if set(frozen["models"]) != set(MODEL_NAMES) or set(frozen["calibrations"]) != set(MODEL_NAMES):
        raise ValueError("The freeze must bind exactly three model pipelines")
    if frozen["config"].get("experiment_plan", {}).get("uncertainty") != BOOTSTRAP_PLAN:
        raise ValueError("The exact bootstrap protocol must be predeclared before final evaluation")
    if summary.get("schema_version") != 1 or summary.get("status") != "completed" or summary.get("final_outcomes_used_for_selection") is not False:
        raise ValueError("A completed release without final-outcome reselection is required")
    if summary.get("freeze_sha256") != frozen["freeze_sha256"] or summary.get("manifest_sha256") != manifest["manifest_sha256"]:
        raise ValueError("Release evidence belongs to a different freeze or manifest")
    if summary.get("counts") != manifest["counts"] or summary.get("calibrations") != frozen["calibrations"] or summary.get("selection") != frozen["config"].get("selection"):
        raise ValueError("Released counts, calibration or selection differs from the freeze")
    selection = summary["selection"]
    final_selection = selection.get("calibrated_pipeline_selection", {})
    if selection.get("selection_split") != "selection" or final_selection.get("decision_data") != "selection" or final_selection.get("final_outcomes_used") is not False:
        raise ValueError("Pipeline decisions must precede final evaluation and use selection only")
    if summary.get("selected_transformer") != frozen["config"].get("selected_transformer") or summary["selected_transformer"] != selection.get("selected_model"):
        raise ValueError("Released selected pipeline differs from the freeze")
    if frozen["config"]["experiment_plan"].get("selection", {}).get("final_holdout_reselects_models") is not False:
        raise ValueError("The frozen plan must prohibit final-outcome reselection")
    if set(summary.get("models", {})) != set(MODEL_NAMES):
        raise ValueError("All three final reports are required")
    records = [r for r in manifest["records"] if r["split"] == "locked_holdout"]
    minimal = [{k: r[k] for k in ("id", "label", "text_1_sha256", "text_2_sha256", "group_id")} for r in records]
    predictions = {}
    evidence = {}
    for name in MODEL_NAMES:
        directory = Path(prediction_root) / name
        metrics_path, predictions_path = directory / "metrics.json", directory / "predictions.jsonl"
        if directory.is_symlink() or predictions_path.is_symlink() or not predictions_path.is_file():
            raise ValueError("Regular prediction evidence is required for " + name)
        metrics = _read_json(metrics_path)
        if metrics != summary["models"][name]:
            raise ValueError("Published metrics differ from the release summary for " + name)
        if metrics.get("model_name") != name or metrics.get("split") != "locked_holdout" or metrics.get("freeze_sha256") != frozen["freeze_sha256"] or metrics.get("manifest_sha256") != manifest["manifest_sha256"]:
            raise ValueError("Prediction report scope or freeze differs for " + name)
        if metrics.get("source_sha256") != frozen["source_sha256"] or metrics.get("evaluation_records_sha256") != canonical_sha256(minimal):
            raise ValueError("Prediction source or ordered record binding differs for " + name)
        if metrics.get("model_content_sha256") != frozen["models"][name]["content_sha256"] or metrics.get("model_artifacts") != frozen["models"][name]["files"] or metrics.get("calibration_sha256") != canonical_sha256(frozen["calibrations"][name]):
            raise ValueError("Prediction model or calibration binding differs for " + name)
        if metrics.get("predictions_sha256") != file_sha256(predictions_path):
            raise ValueError("Declared prediction SHA256 mismatch for " + name)
        rows = [_parse_json(line) for line in predictions_path.read_text(encoding="utf-8").splitlines()]
        if len(rows) != len(records):
            raise ValueError("Prediction IDs do not cover every final pair for " + name)
        calibration = frozen["calibrations"][name]
        predicted = []
        for expected, row in zip(records, rows):
            if type(row.get("label")) is not int or any(row.get(key) != expected[key] for key in ("id", "label", "group_id", "split")):
                raise ValueError("Prediction ID/label/group alignment differs for " + name)
            probability = row.get("calibrated_probability")
            label = row.get("prediction")
            if isinstance(probability, bool) or not isinstance(probability, (float, int)) or not math.isfinite(probability) or not 0 <= probability <= 1:
                raise ValueError("Invalid calibrated probability for " + name)
            if type(label) is not int or label not in (0, 1) or label != int(probability >= calibration["threshold"]):
                raise ValueError("Prediction differs from its frozen threshold for " + name)
            confidence = probability if label else 1 - probability
            if type(row.get("accepted")) is not bool or row["accepted"] != (confidence >= calibration["abstention"]["confidence_threshold"]):
                raise ValueError("Abstention differs from the frozen policy for " + name)
            predicted.append(label)
        y, p = np.array([r["label"] for r in records]), np.array(predicted)
        confusion = np.bincount(y * 2 + p, minlength=4)
        points = _confusion_metrics(confusion)
        if metrics.get("count") != len(records) or metrics.get("correct") != int(confusion[0] + confusion[3]) or metrics.get("confusion") != confusion.reshape(2, 2).tolist():
            raise ValueError("Recorded count/correct/confusion differs for " + name)
        for metric in BOOTSTRAP_PLAN["metrics"]:
            _close(float(points[metric]), metrics.get(metric), "Recorded " + metric + " differs for " + name)
        predictions[name] = predicted
        evidence[name] = {"predictions_sha256": metrics["predictions_sha256"], "metrics_file_sha256": file_sha256(metrics_path),
                          "model_content_sha256": metrics["model_content_sha256"], "calibration_sha256": metrics["calibration_sha256"]}
    result = paired_group_bootstrap([r["label"] for r in records], predictions, [r["group_id"] for r in records])
    result.update({"schema_version": 1, "status": "completed", "protocol": BOOTSTRAP_PLAN, "split": "locked_holdout",
                   "freeze_sha256": frozen["freeze_sha256"], "manifest_sha256": manifest["manifest_sha256"],
                   "manifest_file_sha256": file_sha256(manifest_path), "freeze_file_sha256": file_sha256(freeze_path),
                   "release_summary_file_sha256": file_sha256(summary_path), "predictions_evidence": evidence,
                   "source_sha256": frozen["source_sha256"], "classification_scope": "Every final pair, independently of abstention.",
                   "interval_method": "Empirical 2.5th/97.5th percentiles across 1,000 paired whole-component bootstrap draws; linear interpolation.",
                   "model_inference_performed": False, "parameters_refitted": False, "configuration_reselected": False,
                   "training_seed_count": 1, "training_seed_uncertainty_estimated": False,
                   "author_or_domain_generalization_uncertainty_estimated": False,
                   "interpretation": "Conditional on this provided corpus and its observed document components for one trained seed. This is not an unseen-author, unseen-domain or repeated-training confidence interval. Group exchangeability remains an assumption; historical data exposure and near-duplicate dependence are not removed by bootstrap."})
    if source_inventory() != frozen["source_sha256"]:
        raise ValueError("Bound executable source changed during bootstrap reporting")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True, help="locked-holdout directory containing three model subdirectories")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = bootstrap_results(args.manifest, args.freeze, args.summary, args.prediction_root, args.output)
    print(json.dumps({"status": result["status"], "output": str(args.output), "freeze_sha256": result["freeze_sha256"],
                      "pairs": result["evaluated_pair_count"], "groups": result["document_component_count"]}))


if __name__ == "__main__":
    main()
