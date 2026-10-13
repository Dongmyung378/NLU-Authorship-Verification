import math
import unittest

from nlu_authorship.metrics import binary_metrics, sigmoid, swapped_diagnostics, confidence_accuracy_bins


class MetricsTests(unittest.TestCase):
    def test_known_confusion_auc_and_fixed_bins(self):
        report = binary_metrics([0, 0, 1, 1], [.1, .4, .35, .8])
        self.assertEqual(report["confusion"], [[2, 0], [1, 1]])
        self.assertEqual(report["accuracy"], .75)
        self.assertAlmostEqual(report["roc_auc"], .75)
        self.assertAlmostEqual(report["macro_f1"], (.8 + 2 / 3) / 2)
        self.assertAlmostEqual(report["brier_score"], (.01 + .16 + .4225 + .04) / 4)
        self.assertEqual(sum(x["count"] for x in report["ece10_bins"]), 4)

    def test_tied_auc_and_unavailable_class_are_explicit(self):
        self.assertEqual(binary_metrics([0, 1], [.5, .5])["roc_auc"], .5)
        self.assertIsNone(binary_metrics([1, 1], [.1, .9])["roc_auc"])
        self.assertTrue(math.isfinite(binary_metrics([0, 1], [0, 1])["log_loss"]))

    def test_invalid_scores_and_swapped_mismatch(self):
        with self.assertRaises(ValueError):
            binary_metrics([0, 1], [float("nan"), .5])
        self.assertTrue((sigmoid([-1000, 1000]) > 0).all())
        report = swapped_diagnostics([[0, -2], [0, 2]], [[0, 2], [0, -2]])
        self.assertEqual(report["prediction_mismatch_count"], 2)

    def test_confidence_bin_accuracy_is_not_positive_label_frequency(self):
        table = confidence_accuracy_bins([0, 1], [.1, .9])
        self.assertEqual(len(table), 10)
        self.assertEqual(table[9]["count"], 2)
        self.assertEqual(table[9]["correct"], 2)
        self.assertEqual(table[9]["accuracy"], 1)
        self.assertAlmostEqual(table[9]["mean_confidence"], .9)
        self.assertIsNone(table[0]["accuracy"])
        self.assertIsNone(table[0]["mean_confidence"])

    def test_frozen_threshold_controls_predicted_label_confidence(self):
        table = confidence_accuracy_bins([0, 1], [.6, .7], threshold=.65)
        self.assertEqual(table[4]["count"], 1)
        self.assertEqual(table[7]["count"], 1)
        self.assertEqual(sum(row["correct"] for row in table), 2)
