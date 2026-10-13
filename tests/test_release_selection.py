"""Calibration-fixed deployment selection is separate from raw loss analysis."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from nlu_authorship.calibration import fit_calibration, select_model_policy
from nlu_authorship.release import release_models, select_calibrated_pipeline


def logits(margins):
    margins = np.asarray(margins, dtype=float)
    return np.stack([np.zeros_like(margins), margins], axis=1)


def comparisons(ce, asl, labels=None):
    labels = list(labels) if labels is not None else [0, 0, 1, 1]
    return {name: {'ids': ['selection-'+str(i) for i in range(len(labels))],
                   'labels': labels, 'logits_ab': logits(values), 'logits_ba': logits(values)}
            for name, values in [('ce', ce), ('custom_asl', asl), ('stylometric', ce)]}


def calibration(ce=(-4, -2, 2, 4), asl=(2, 2.2, 3, 4)):
    return {name: fit_calibration(logits(values), [0, 0, 1, 1], ids=['calibration-'+str(i) for i in range(4)])
            for name, values in [('ce', ce), ('custom_asl', asl), ('stylometric', ce)]}


def identity_calibration():
    return {'schema_version': 1, 'fitting_split': 'calibration', 'inference_method': 'ab',
            'temperature': 1., 'threshold': .5,
            'abstention': {'confidence_threshold': .5, 'target_coverage': .8,
                           'target_accepted_accuracy': .9}}


class CalibratedReleaseSelectionTests(unittest.TestCase):
    def test_calibration_can_select_asl_while_raw_threshold_selects_ce(self):
        inputs = comparisons([-3, 1, 2, 4], [2.1, 2.4, 3.5, 4.2])
        raw = select_model_policy(inputs)
        fitted = calibration()
        raw_before, fitted_before = copy.deepcopy(raw), copy.deepcopy(fitted)
        self.assertEqual(raw['selected_model'], 'ce')
        with patch('nlu_authorship.calibration.fit_calibration', side_effect=AssertionError('Do not refit during selection')):
            result = select_calibrated_pipeline(inputs, fitted, raw)
        self.assertEqual(result['selected_model'], 'custom_asl')
        self.assertEqual(result['raw_selected_model'], 'ce')
        self.assertEqual(result['raw_selection'], raw_before)
        self.assertEqual(result['decision_threshold'], fitted['custom_asl']['threshold'])
        self.assertEqual(result['calibrated_pipeline_selection']['outcomes']['custom_asl']['metrics']['macro_f1'], 1.)
        self.assertTrue(result['calibrated_pipeline_selection']['alternative_adoption_gate_passed'])
        self.assertEqual(raw, raw_before)
        self.assertEqual(fitted, fitted_before)

    def test_calibration_can_keep_ce_despite_raw_asl_winner(self):
        inputs = comparisons([2.1, 2.4, 3.5, 4.2], [-3, 1, 2, 4])
        raw = select_model_policy(inputs)
        self.assertEqual(raw['selected_model'], 'custom_asl')
        fitted = calibration(ce=(2, 2.2, 3, 4), asl=(100, 101, 102, 103))
        result = select_calibrated_pipeline(inputs, fitted, raw)
        self.assertEqual(result['raw_selected_model'], 'custom_asl')
        self.assertEqual(result['selected_model'], 'ce')
        self.assertLess(result['calibrated_pipeline_selection']['alternative_macro_f1_gain'], 0)

    def test_gate_rejects_small_gain_and_accepts_larger_gain(self):
        for count, expected in [(1000, 'ce'), (400, 'custom_asl')]:
            y = [0]*(count//2)+[1]*(count//2)
            perfect = [-2.]*(count//2)+[2.]*(count//2)
            imperfect = perfect.copy()
            imperfect[-1] = -2.
            inputs = comparisons(imperfect, perfect, y)
            fitted = {name: identity_calibration() for name in inputs}
            result = select_calibrated_pipeline(inputs, fitted, select_model_policy(inputs))
            self.assertEqual(result['selected_model'], expected)
            gain = result['calibrated_pipeline_selection']['alternative_macro_f1_gain']
            self.assertEqual(gain >= .002, expected == 'custom_asl')

    def test_exact_tie_defaults_to_ce(self):
        inputs = comparisons([-2, -1, 1, 2], [-2, -1, 1, 2])
        fitted = {name: identity_calibration() for name in inputs}
        result = select_calibrated_pipeline(inputs, fitted, select_model_policy(inputs))
        self.assertEqual(result['selected_model'], 'ce')
        self.assertEqual(result['alternative_macro_f1_gain'], 0.)
        self.assertFalse(result['calibrated_pipeline_selection']['alternative_adoption_gate_passed'])

    def test_invalid_parameters_or_scope_reject_before_decision(self):
        inputs = comparisons([-2, -1, 1, 2], [-2, -1, 1, 2])
        raw = select_model_policy(inputs)
        fitted = {name: identity_calibration() for name in inputs}
        for split in ('train', 'calibration', 'locked_holdout'):
            with self.assertRaises(ValueError):
                select_calibrated_pipeline(inputs, fitted, raw, split=split)
        for gate in (.001, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                select_calibrated_pipeline(inputs, fitted, raw, minimum_improvement=gate)
        for field in ('temperature', 'threshold'):
            bad = copy.deepcopy(fitted)
            bad['custom_asl'][field] = float('nan')
            with self.assertRaises(ValueError):
                select_calibrated_pipeline(inputs, bad, raw)
        bad_inputs = copy.deepcopy(inputs)
        bad_inputs['custom_asl']['logits_ab'][0, 1] = float('nan')
        with self.assertRaises(ValueError):
            select_calibrated_pipeline(bad_inputs, fitted, raw)

    def test_record_alignment_and_fixed_orientation_are_required(self):
        inputs = comparisons([-2, -1, 1, 2], [-2, -1, 1, 2])
        raw = select_model_policy(inputs)
        fitted = {name: identity_calibration() for name in inputs}
        bad = copy.deepcopy(inputs)
        bad['custom_asl']['ids'][0] = 'locked-holdout-record'
        with self.assertRaises(ValueError):
            select_calibrated_pipeline(bad, fitted, raw)
        bad_calibration = copy.deepcopy(fitted)
        bad_calibration['custom_asl']['inference_method'] = 'symmetric'
        with self.assertRaises(ValueError):
            select_calibrated_pipeline(inputs, bad_calibration, raw)

    def test_all_three_calibrations_precede_final_selection_and_freeze(self):
        inputs = comparisons([-3, 1, 2, 4], [2.1, 2.4, 3.5, 4.2])
        fitted = calibration()
        selection_records = [{'id': i, 'label': y} for i, y in zip(inputs['ce']['ids'], inputs['ce']['labels'])]
        runners = {name: object() for name in inputs}
        names = {id(runner): name for name, runner in runners.items()}
        fitted_names, prediction_calls = [], []
        state = {'frozen': False}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan = root/'plan.json'
            plan.write_text(json.dumps({'selection': {'asymmetric_loss_min_macro_f1_gain': .002},
                                        'calibration': {'temperature_bounds': [.05, 20.],
                                                        'abstention_target_minimum_coverage': .8,
                                                        'abstention_target_minimum_accepted_accuracy': .9}}))
            output = root/'release'

            def predictions(runner, records, *, orientation, batch_size):
                self.assertEqual(records, selection_records)
                name = names[id(runner)]
                prediction_calls.append((name, orientation))
                return inputs[name]['logits_'+orientation]

            def fit(runner, repo, manifest, path, *, method, **kwargs):
                self.assertFalse((output/'selection.json').exists())
                name = names[id(runner)]
                fitted_names.append(name)
                self.assertEqual(method, 'ab')
                return fitted[name]

            def freeze(models, manifest, config, calibrations, path, source_root):
                self.assertEqual(set(fitted_names), set(inputs))
                self.assertEqual(config['selected_transformer'], 'custom_asl')
                saved = json.loads((output/'selection.json').read_text())
                self.assertEqual(saved['selected_model'], 'custom_asl')
                self.assertEqual(saved['raw_selected_model'], 'ce')
                self.assertEqual(calibrations, fitted)
                state['frozen'] = True
                return {'freeze_sha256': 'f'*64}

            def evaluate(runner, repo, manifest, split, output_path, params, model, name, **kwargs):
                self.assertTrue(state['frozen'])
                self.assertEqual(split, 'locked_holdout')
                self.assertEqual(params, fitted[name])
                return {'count': 4}

            with patch('nlu_authorship.release.read_manifest', return_value={'manifest_sha256': 'm'*64, 'counts': {'selection': 4}}), \
                    patch('nlu_authorship.release.load_records', return_value=selection_records) as load, \
                    patch('nlu_authorship.release.runner_for', side_effect=lambda name, path, device: runners[name]), \
                    patch('nlu_authorship.release.predict_records_logits', side_effect=predictions), \
                    patch('nlu_authorship.release.fit_model_calibration', side_effect=fit), \
                    patch('nlu_authorship.release.freeze_evaluation', side_effect=freeze), \
                    patch('nlu_authorship.release.evaluate_model', side_effect=evaluate) as final:
                result = release_models(root, root/'manifest.json', {name: root/name for name in inputs},
                                        plan, output, device='cpu')
            self.assertEqual(result['selected_transformer'], 'custom_asl')
            self.assertEqual(len(prediction_calls), 6)
            self.assertEqual(final.call_count, 3)
            self.assertEqual([call.args[2] for call in load.call_args_list], ['selection'])
            self.assertFalse(result['final_outcomes_used_for_selection'])
