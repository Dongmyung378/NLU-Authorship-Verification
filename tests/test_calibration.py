import unittest
import numpy as np

from nlu_authorship.calibration import (apply_calibration, fit_abstention, fit_calibration,
                                        fit_temperature, fit_threshold, select_model_policy)


class CalibrationTests(unittest.TestCase):
    def test_temperature_improves_overconfident_nll_and_honest_scope(self):
        logits = np.array([[0, -8], [0, -6], [0, 6], [0, 8]], dtype=float)
        labels = [0, 1, 0, 1]
        result = fit_calibration(logits, labels, ids=["a", "b", "c", "d"])
        self.assertGreater(result["temperature"], 1)
        self.assertLessEqual(result["temperature_fit"]["fitted_calibration_nll"], result["temperature_fit"]["identity_calibration_nll"])
        self.assertEqual(result["fitting_split"], "calibration")
        self.assertEqual(len(apply_calibration(logits, result)), 4)

    def test_all_fitting_rejects_selection_or_holdout(self):
        for fn, args in [(fit_temperature, ([1, -1], [1, 0])),
                         (fit_threshold, ([.9, .1], [1, 0])), (fit_abstention, ([.9, .1], [1, 0]))]:
            with self.assertRaisesRegex(ValueError, "calibration only"):
                fn(*args, split="locked_holdout")

    def test_abstention_target_failure_is_reported(self):
        impossible = fit_abstention([.5] * 4, [0, 1, 0, 1])
        self.assertFalse(impossible["target_achieved_on_calibration"])
        self.assertEqual(impossible["coverage"], 1)
        self.assertEqual(impossible["accepted_accuracy"], .5)
        possible = fit_abstention([.01, .1, .9, .99, .55], [0, 0, 1, 1, 0])
        self.assertTrue(possible["target_achieved_on_calibration"])
        self.assertEqual(possible["accepted_count"], 4)

    def test_symmetric_policy_no_harm_and_raw_ab_loss_comparison(self):
        inputs = {"ids": ["a", "b", "c", "d"], "labels": [0, 0, 1, 1],
                  "logits_ab": [[0, -2], [0, -2], [0, 2], [0, 2]],
                  "logits_ba": [[0, -1], [0, -1], [0, 1], [0, 1]]}
        result = select_model_policy({"ce": inputs, "custom_asl": inputs})
        self.assertEqual(result["selected_model"], "ce")
        self.assertEqual(result["selected_method"], "symmetric")
        self.assertEqual(result["alternative_macro_f1_gain_raw_ab"], 0)
        self.assertTrue(result["outcomes"]["ce"]["symmetric_invariance_no_harm_gate_passed"])
        self.assertEqual(result["outcomes"]["ce"]["symmetric_max_probability_gap"], 0)

    def test_calibration_rejects_invalid_cutoff_before_use(self):
        result = fit_calibration([[0, -1], [0, 1]], [0, 1])
        result["abstention"]["confidence_threshold"] = float("nan")
        with self.assertRaises(ValueError):
            apply_calibration([[0, 0]], result)
