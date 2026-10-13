import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np

from nlu_authorship.calibration import fit_calibration
from nlu_authorship.data import build_split_manifest, load_records
from nlu_authorship.evaluation import (evaluate_model, fit_model_calibration,
                                       freeze_evaluation, validate_freeze)
from nlu_fixtures import write_artifact, write_dataset, write_source


class Runner:
    def __init__(self):
        self.calls = 0

    def predict_logits(self, records, batch_size=16, orientation="ab"):
        self.calls += 1
        margins = np.array([2 if index % 2 else -2 for index in range(len(records))])
        if orientation == "ba":
            margins = -margins
        return np.stack([np.zeros_like(margins), margins], axis=1)


class EvaluationProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.repo = write_dataset(root / "repo")
        self.source = write_source(root / "source")
        self.model = write_artifact(root / "model")
        self.manifest = build_split_manifest(self.repo)
        self.runner = Runner()
        self.calibration = fit_model_calibration(self.runner, self.repo, self.manifest, self.model)
        self.freeze = root / "freeze.json"
        freeze_evaluation({"ce": self.model}, self.manifest, {"selection_split": "selection"}, {"ce": self.calibration}, self.freeze, self.source)
        self.root = root

    def tearDown(self):
        self.temp.cleanup()

    def test_holdout_requires_freeze_before_runner_calls(self):
        runner = Runner()
        with self.assertRaisesRegex(ValueError, "freeze"):
            evaluate_model(runner, self.repo, self.manifest, "locked_holdout", self.root / "no-freeze", self.calibration, self.model, "ce")
        self.assertEqual(runner.calls, 0)

    def test_source_artifact_and_calibration_mutation_rejected_before_inference(self):
        bad = copy.deepcopy(self.calibration)
        bad["temperature"] *= 2
        runner = Runner()
        with self.assertRaisesRegex(ValueError, "differs from freeze"):
            evaluate_model(runner, self.repo, self.manifest, "locked_holdout", self.root / "bad-calibration", bad, self.model, "ce", self.freeze, source_root=self.source)
        self.assertEqual(runner.calls, 0)
        (self.model / "weights.bin").write_bytes(b"different")
        with self.assertRaisesRegex(ValueError, "checksum"):
            validate_freeze(self.freeze, self.manifest, self.model, "ce", self.source)
        (self.model / "weights.bin").write_bytes(b"saved-test-weights")
        (self.source / "nlu_authorship" / "bound.py").write_text("VALUE=2\n")
        with self.assertRaisesRegex(ValueError, "source"):
            validate_freeze(self.freeze, self.manifest, self.model, "ce", self.source)

    def test_portable_model_path_and_complete_ordered_outputs(self):
        moved = self.root / "runtime" / "models" / "ce"
        shutil.copytree(self.model, moved)
        validate_freeze(self.freeze, self.manifest, moved, "ce", self.source)
        output = self.root / "evaluation"
        metrics = evaluate_model(self.runner, self.repo, self.manifest, "locked_holdout", output, self.calibration, moved, "ce", self.freeze, source_root=self.source)
        records = load_records(self.repo, self.manifest, "locked_holdout")
        predictions = [json.loads(row) for row in (output / "predictions.jsonl").read_text().splitlines()]
        self.assertEqual(metrics["count"], len(records))
        self.assertEqual([row["id"] for row in predictions], [r["id"] for r in records])
        self.assertTrue(all("text_1" not in row and "text_2" not in row for row in predictions))
        self.assertEqual(sum(row["count"] for row in metrics["length_bins"].values()), len(records))
        self.assertFalse(metrics["author_or_domain_separation_verified"])
        self.assertEqual(metrics["abstention"]["parameters_fitted_on"], "calibration")
        for key in ("raw_selected_confidence_accuracy_bins", "calibrated_confidence_accuracy_bins"):
            self.assertEqual(sum(row["count"] for row in metrics[key]), len(records))
        self.assertEqual(sum(row["correct"] for row in metrics["calibrated_confidence_accuracy_bins"]), metrics["correct"])

    def test_symmetric_selected_policy_is_invariant_while_raw_orders_differ(self):
        calibration = fit_model_calibration(self.runner, self.repo, self.manifest, self.model, method="symmetric")
        freeze = self.root / "symmetric-freeze.json"
        freeze_evaluation({"ce": self.model}, self.manifest, {}, {"ce": calibration}, freeze, self.source)
        metrics = evaluate_model(self.runner, self.repo, self.manifest, "locked_holdout", self.root / "symmetric-evaluation",
                                 calibration, self.model, "ce", freeze, source_root=self.source)
        self.assertGreater(metrics["swapped_input_diagnostics_raw"]["max_absolute_probability_gap"], 0)
        self.assertEqual(metrics["selected_policy_swapped_input_diagnostics"]["max_absolute_probability_gap"], 0)
        self.assertEqual(metrics["selected_policy_swapped_input_diagnostics"]["prediction_mismatch_count"], 0)
