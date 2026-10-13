"""Bind saved artifacts and calibration before held-out model inference."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .calibration import apply_calibration, fit_calibration, selected_margins, validate_calibration
from .data import (canonical_sha256, file_sha256, load_records, read_manifest,
                   validate_manifest)
from .metrics import (binary_metrics, logit_margins, probability_logits, sigmoid,
                      swapped_diagnostics, confidence_accuracy_bins)
from .normalization import length_bin, minimum_word_count

CLASS_NAMES = ["Different authors", "Same author"]


def artifact_inventory(path) -> dict:
    path = Path(path)
    if path.is_symlink():
        raise ValueError("Model artifact symlinks are forbidden")
    path = path.resolve(strict=True)
    kind = "directory" if path.is_dir() else "file"
    files = sorted(path.rglob("*")) if path.is_dir() else [path]
    inventory = {}
    for item in files:
        if item.is_symlink():
            raise ValueError("Model artifact symlinks are forbidden")
        if not item.is_file():
            continue
        if path.is_dir() and not item.resolve().is_relative_to(path):
            raise ValueError("Artifact leaves model directory")
        relative = item.relative_to(path).as_posix() if path.is_dir() else item.name
        inventory[relative] = {"sha256": file_sha256(item), "bytes": item.stat().st_size}
    if not inventory:
        raise ValueError("Model artifact inventory is empty")
    return {"kind": kind, "files": inventory, "content_sha256": canonical_sha256(inventory)}


def source_inventory(source_root=None) -> dict:
    root = Path(source_root) if source_root is not None else Path(__file__).resolve().parents[1]
    paths = set((root / "nlu_authorship").rglob("*.py"))
    paths.update((root / "scripts").rglob("*.py"))
    paths.update((root / "scripts").rglob("*.ps1"))
    paths.update((root / "configs").rglob("*.json"))
    paths.update(root.glob("requirements*.txt"))
    paths.update(root.glob("launch-nlu.*"))
    if (root / "run.ps1").is_file():
        paths.add(root / "run.ps1")
    if (root / "pyproject.toml").is_file():
        paths.add(root / "pyproject.toml")
    if not paths:
        raise ValueError("Source inventory is empty")
    return {path.relative_to(root).as_posix(): file_sha256(path) for path in sorted(paths) if path.is_file()}


def calibration_records_sha256(manifest) -> str:
    manifest = read_manifest(manifest)
    return canonical_sha256([{"id": r["id"], "label": r["label"]}
                             for r in manifest["records"] if r["split"] == "calibration"])


def freeze_evaluation(models, manifest, config, calibrations, output_path, source_root=None):
    manifest = read_manifest(manifest)
    validate_manifest(manifest)
    if not models or set(calibrations) != set(models):
        raise ValueError("Every frozen model needs its own fitted calibration")
    model_inventory = {name: artifact_inventory(path) for name, path in models.items()}
    expected_calibration = calibration_records_sha256(manifest)
    for name, calibration in calibrations.items():
        if calibration.get("fitting_split") != "calibration" or calibration.get("count") != manifest["counts"]["calibration"]:
            raise ValueError("Calibration scope/count differs from the declared calibration split")
        if calibration.get("calibration_records_sha256") != expected_calibration:
            raise ValueError("Calibration report does not bind the exact ordered calibration records")
        if calibration.get("model_content_sha256") is not None and calibration["model_content_sha256"] != model_inventory[name]["content_sha256"]:
            raise ValueError("Calibration was fitted for a different saved artifact")
        if calibration.get("split_manifest_sha256") is not None and calibration["split_manifest_sha256"] != manifest["manifest_sha256"]:
            raise ValueError("Calibration was fitted for a different dataset split")
        apply_calibration([[0.0, 0.0]], calibration, [[0.0, 0.0]])
    frozen = {"schema_version": 1, "protocol": manifest["protocol"],
              "manifest_sha256": manifest["manifest_sha256"], "full_manifest_sha256": canonical_sha256(manifest),
              "source_csvs": manifest["source_csvs"], "models": model_inventory,
              "config": config, "config_sha256": canonical_sha256(config),
              "calibrations": calibrations, "calibrations_sha256": canonical_sha256(calibrations),
              "source_sha256": source_inventory(source_root), "class_names": CLASS_NAMES,
              "class_ids": [0, 1], "final_evaluation_split": "locked_holdout",
              "historical_exposure": manifest["historical_exposure"],
              "author_or_domain_separation_verified": False,
              "selection_scope": "selection chooses model/orientation; calibration fits probability/threshold/abstention; final outcomes never reselect."}
    frozen["freeze_sha256"] = canonical_sha256(frozen)
    output = Path(output_path)
    if output.exists():
        raise FileExistsError("Evaluation freeze is immutable; choose a new output path")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(frozen, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    return frozen


def validate_freeze(freeze_path, manifest, model_path=None, model_name=None, source_root=None):
    frozen = json.loads(Path(freeze_path).read_text(encoding="utf-8"))
    manifest = read_manifest(manifest)
    validate_manifest(manifest)
    if frozen.get("schema_version") != 1 or frozen.get("freeze_sha256") != canonical_sha256({k: v for k, v in frozen.items() if k != "freeze_sha256"}):
        raise ValueError("Freeze checksum mismatch")
    if frozen.get("manifest_sha256") != manifest["manifest_sha256"] or frozen.get("full_manifest_sha256") != canonical_sha256(manifest):
        raise ValueError("Frozen dataset split differs")
    if frozen.get("config_sha256") != canonical_sha256(frozen["config"]) or frozen.get("calibrations_sha256") != canonical_sha256(frozen["calibrations"]):
        raise ValueError("Frozen configuration or calibration differs")
    if frozen.get("source_sha256") != source_inventory(source_root):
        raise ValueError("Executable source/configuration/dependencies changed after freeze")
    if frozen.get("class_names") != CLASS_NAMES or frozen.get("class_ids") != [0, 1]:
        raise ValueError("Frozen class mapping differs")
    if model_name is not None and model_name not in frozen["models"]:
        raise ValueError("Model name is absent from freeze")
    if model_path is not None:
        actual = artifact_inventory(model_path)
        expected = [frozen["models"][model_name]] if model_name is not None else list(frozen["models"].values())
        if actual not in expected:
            raise ValueError("Saved model/tokenizer/config artifact checksum differs from freeze")
    return frozen


def predict_records_logits(runner, records, *, orientation="ab", batch_size=16):
    if orientation not in {"ab", "ba"} or batch_size < 1:
        raise ValueError("Invalid inference orientation or batch size")
    if hasattr(runner, "predict_logits"):
        logits = np.asarray(runner.predict_logits(records, batch_size=batch_size, orientation=orientation), dtype=np.float64)
    elif hasattr(runner, "predict_proba"):
        logits = probability_logits(runner.predict_proba(records, batch_size=batch_size, orientation=orientation))
    else:
        raise TypeError("Runner must implement predict_logits or predict_proba")
    if len(logits) != len(records):
        raise ValueError("Runner predictions do not cover the ordered records")
    logit_margins(logits)
    return logits


def fit_model_calibration(runner, repo_root, manifest, model_path, *, method="ab", batch_size=16, **kwargs):
    records = load_records(repo_root, manifest, "calibration")
    before = artifact_inventory(model_path)
    ab = predict_records_logits(runner, records, orientation="ab", batch_size=batch_size)
    ba = predict_records_logits(runner, records, orientation="ba", batch_size=batch_size)
    if artifact_inventory(model_path) != before:
        raise ValueError("Model changed during calibration inference")
    report = fit_calibration(ab, [r["label"] for r in records], ba, method=method,
                             ids=[r["id"] for r in records], **kwargs)
    report["model_content_sha256"] = before["content_sha256"]
    report["split_manifest_sha256"] = read_manifest(manifest)["manifest_sha256"]
    return report


def _report_predictions(records, logits_ab, logits_ba, calibration, split):
    labels = np.asarray([r["label"] for r in records], dtype=np.int64)
    p_ab, p_ba = sigmoid(logit_margins(logits_ab)), sigmoid(logit_margins(logits_ba))
    p_symmetric = sigmoid(selected_margins(logits_ab, logits_ba, "symmetric"))
    raw = sigmoid(selected_margins(logits_ab, logits_ba, calibration["inference_method"]))
    probabilities = apply_calibration(logits_ab, calibration, logits_ba)
    swapped_selected = apply_calibration(logits_ba, calibration, logits_ab)
    threshold = calibration["threshold"]
    predictions = (probabilities >= threshold).astype(int)
    confidence = np.where(predictions == 1, probabilities, 1 - probabilities)
    cutoff = float(calibration["abstention"]["confidence_threshold"])
    if not np.isfinite(cutoff) or not 0 <= cutoff <= 1:
        raise ValueError("Invalid abstention cutoff")
    accepted = confidence >= cutoff
    count = int(accepted.sum())
    accuracy = float(np.mean(predictions[accepted] == labels[accepted])) if count else None
    rows = []
    for i, record in enumerate(records):
        rows.append({"id": record["id"], "label": record["label"], "group_id": record["group_id"],
                     "split": split, "raw_probability_ab": float(p_ab[i]), "raw_probability_ba": float(p_ba[i]),
                     "raw_probability_symmetric": float(p_symmetric[i]), "raw_probability_selected": float(raw[i]),
                     "calibrated_probability": float(probabilities[i]), "prediction": int(predictions[i]),
                     "calibrated_probability_swapped_selected_policy": float(swapped_selected[i]),
                     "confidence": float(confidence[i]), "accepted": bool(accepted[i]),
                     "inference_method": calibration["inference_method"],
                     "minimum_cleaned_words": minimum_word_count(record["text_1"], record["text_2"]),
                     "length_bin": length_bin(record["text_1"], record["text_2"])})
    lengths = {}
    for name in ("under_50", "50_to_199", "200_plus"):
        mask = np.asarray([row["length_bin"] == name for row in rows])
        lengths[name] = binary_metrics(labels[mask], probabilities[mask], threshold) if mask.any() else {"count": 0, "accuracy": None, "macro_f1": None}
    metrics = binary_metrics(labels, probabilities, threshold)
    selected_gap = np.abs(probabilities - swapped_selected)
    selected_mismatch = (probabilities >= threshold) != (swapped_selected >= threshold)
    metrics.update({"raw_metrics": binary_metrics(labels, raw), "raw_ab_metrics": binary_metrics(labels, p_ab),
                    "raw_ba_metrics": binary_metrics(labels, p_ba), "raw_symmetric_metrics": binary_metrics(labels, p_symmetric),
                    "raw_selected_confidence_accuracy_bins": confidence_accuracy_bins(labels, raw, .5),
                    "calibrated_confidence_accuracy_bins": confidence_accuracy_bins(labels, probabilities, threshold),
                    "confidence_accuracy_bin_policy": "Ten fixed equal-width predicted-label confidence bins: p for selected label1,1-p for selected label0; accuracy is observed correct/count. Raw threshold0.5; calibrated threshold frozen from calibration. Empty accuracy/mean confidence=null. These are not positive-class frequency bins.",
                    "swapped_input_diagnostics_raw": swapped_diagnostics(logits_ab, logits_ba),
                    "swapped_input_diagnostics_calibrated": swapped_diagnostics(logits_ab, logits_ba, temperature=calibration["temperature"], threshold=threshold),
                    "selected_policy_swapped_input_diagnostics": {"count": len(labels), "inference_method": calibration["inference_method"],
                        "mean_absolute_probability_gap": float(selected_gap.mean()), "p95_absolute_probability_gap": float(np.percentile(selected_gap, 95)),
                        "max_absolute_probability_gap": float(selected_gap.max()), "prediction_mismatch_count": int(selected_mismatch.sum()),
                        "prediction_mismatch_fraction": float(selected_mismatch.mean())},
                    "length_bins": lengths, "length_policy": "Minimum word count of the two NFC/RoBERTa-cleaned passages; <50,50-199,>=200; no token-count claim.",
                    "abstention": {"accepted_count": count, "rejected_count": len(labels) - count, "coverage": count / len(labels),
                                   "accepted_accuracy": accuracy, "confidence_threshold": cutoff,
                                   "target_coverage": calibration["abstention"]["target_coverage"],
                                   "target_accepted_accuracy": calibration["abstention"]["target_accepted_accuracy"],
                                   "target_achieved_on_this_evaluation": bool(count and count / len(labels) >= calibration["abstention"]["target_coverage"]
                                                                             and accuracy >= calibration["abstention"]["target_accepted_accuracy"]),
                                   "parameters_fitted_on": "calibration", "no_refit_on_evaluation": True}})
    return rows, metrics


def evaluate_model(runner, repo_root, manifest, split, output_dir, calibration, model_path,
                   model_name, freeze_path=None, batch_size=16, source_root=None):
    manifest = read_manifest(manifest)
    validate_calibration(calibration)
    frozen = None
    if split == "locked_holdout" and freeze_path is None:
        raise ValueError("Final held-out inference requires the immutable evaluation freeze")
    if freeze_path is not None:
        frozen = validate_freeze(freeze_path, manifest, model_path, model_name, source_root)
        if frozen["calibrations"].get(model_name) != calibration:
            raise ValueError("Evaluation calibration/method differs from freeze")
    before = artifact_inventory(model_path)
    records = load_records(repo_root, manifest, split)
    ab = predict_records_logits(runner, records, orientation="ab", batch_size=batch_size)
    ba = predict_records_logits(runner, records, orientation="ba", batch_size=batch_size)
    rows, metrics = _report_predictions(records, ab, ba, calibration, split)
    if artifact_inventory(model_path) != before:
        raise ValueError("Artifact changed during inference")
    if frozen:
        validate_freeze(freeze_path, manifest, model_path, model_name, source_root)
    minimal = [{k: r[k] for k in ("id", "label", "text_1_sha256", "text_2_sha256", "group_id")}
               for r in manifest["records"] if r["split"] == split]
    metrics.update({"schema_version": 1, "model_name": model_name, "model_content_sha256": before["content_sha256"],
                    "model_artifacts": before["files"], "split": split,
                    "manifest_sha256": manifest["manifest_sha256"], "source_csvs": manifest["source_csvs"],
                    "evaluation_records_sha256": canonical_sha256(minimal),
                    "freeze_sha256": frozen["freeze_sha256"] if frozen else None,
                    "source_sha256": source_inventory(source_root), "calibration_sha256": canonical_sha256(calibration),
                    "inference_method": calibration["inference_method"], "class_names": CLASS_NAMES,
                    "historical_exposure": manifest["historical_exposure"],
                    "author_or_domain_separation_verified": False,
                    "scope": "Frozen saved-model evaluation; no training, threshold refit, calibration refit or configuration reselection."})
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    with (output / "predictions.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    metrics["predictions_sha256"] = file_sha256(output / "predictions.jsonl")
    (output / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    (output / "confusion.json").write_text(json.dumps({"class_ids": [0, 1], "class_names": CLASS_NAMES, "rows": "true label", "columns": "predicted label", "matrix": metrics["confusion"]}, indent=2) + "\n", encoding="utf-8", newline="\n")
    return metrics
