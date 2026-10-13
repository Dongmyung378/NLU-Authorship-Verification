import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from nlu_authorship.data import (build_split_manifest, canonical_sha256,
                                 file_sha256, save_manifest)
from nlu_authorship.evaluation import evaluate_model, freeze_evaluation
from nlu_fixtures import write_artifact
from scripts.bootstrap_results import (BOOTSTRAP_PLAN, MODEL_NAMES,
                                       bootstrap_results, group_draw_weights,
                                       paired_group_bootstrap)


class GroupBootstrapTests(unittest.TestCase):
    def test_groups_repeat_whole_and_pair_counts_can_vary(self):
        group_ids = ['a', 'a', 'a', 'b', 'c', 'c']
        groups, positions, weights = group_draw_weights(group_ids)
        self.assertEqual(groups, ['a', 'b', 'c'])
        np.testing.assert_array_equal(weights.sum(axis=1), np.repeat(3, 1000))
        row_weights = weights[:, positions]
        np.testing.assert_array_equal(row_weights[:, 0], row_weights[:, 1])
        np.testing.assert_array_equal(row_weights[:, 1], row_weights[:, 2])
        self.assertGreater(np.ptp(row_weights.sum(axis=1)), 0)
        self.assertFalse(np.array_equal(row_weights[:, 0], row_weights[:, 3]))
        _, _, repeated = group_draw_weights(group_ids)
        np.testing.assert_array_equal(weights, repeated)

    def test_identical_paired_predictions_have_exact_zero_difference(self):
        labels = [0, 1, 0, 1, 1, 0]
        same = [0, 1, 1, 1, 0, 0]
        result = paired_group_bootstrap(labels, dict.fromkeys(MODEL_NAMES, same), ['a', 'a', 'a', 'b', 'c', 'c'])
        self.assertEqual(result['paired_difference']['point'], 0)
        self.assertEqual(result['paired_difference']['lower'], 0)
        self.assertEqual(result['paired_difference']['upper'], 0)
        self.assertAlmostEqual(result['models']['ce']['accuracy']['point'], 4 / 6)
        self.assertEqual(result, paired_group_bootstrap(labels, dict.fromkeys(MODEL_NAMES, same), ['a', 'a', 'a', 'b', 'c', 'c']))

    def test_paired_direction_uses_asl_minus_ce_and_all_rows(self):
        labels = [0, 1, 0, 1]
        result = paired_group_bootstrap(labels, {'ce': [0, 0, 0, 0], 'custom_asl': labels, 'stylometric': labels}, ['a', 'a', 'b', 'b'])
        self.assertAlmostEqual(result['paired_difference']['point'], 2 / 3)
        self.assertEqual(result['models']['custom_asl']['accuracy']['point'], 1)
        with self.assertRaises(ValueError):
            paired_group_bootstrap(labels, {'ce': [0, 0]}, ['a', 'a', 'b', 'b'])


class FrozenBootstrapEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.repo = self.root / 'repo'
        (self.repo / 'data').mkdir(parents=True)
        rows = []
        for group in range(100):
            for member in range(2 + group % 2):
                rows.append({'text_1': f'Anchor document for group {group} shares distinctive prose.',
                             'text_2': f'Partner group {group} observation {member} tells a separate story.',
                             'label': str((group + member) % 2)})
        for name, selected in [('train', rows[:200]), ('dev', rows[200:])]:
            with (self.repo / 'data' / (name + '.csv')).open('w', encoding='utf-8', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=['text_1', 'text_2', 'label'])
                writer.writeheader(); writer.writerows(selected)
        self.manifest = build_split_manifest(self.repo)
        self.manifest_path = self.root / 'manifest.json'
        save_manifest(self.manifest, self.manifest_path)
        calibration_rows = [r for r in self.manifest['records'] if r['split'] == 'calibration']
        calibration = {'schema_version': 1, 'fitting_split': 'calibration', 'count': len(calibration_rows),
                       'calibration_records_sha256': canonical_sha256([{'id': r['id'], 'label': r['label']} for r in calibration_rows]),
                       'inference_method': 'ab', 'temperature': 1.0, 'threshold': .5,
                       'abstention': {'confidence_threshold': .95, 'target_coverage': .8, 'target_accepted_accuracy': .9}}
        self.calibrations = {name: copy.deepcopy(calibration) for name in MODEL_NAMES}
        self.models = {name: write_artifact(self.root / 'model' / name) for name in MODEL_NAMES}
        self.selection = {'selection_split': 'selection', 'selected_model': 'ce',
                          'calibrated_pipeline_selection': {'decision_data': 'selection', 'final_outcomes_used': False}}
        self.config = {'experiment_plan': {'uncertainty': copy.deepcopy(BOOTSTRAP_PLAN), 'selection': {'final_holdout_reselects_models': False}},
                       'selection': self.selection, 'selected_transformer': 'ce'}
        self.freeze_path = self.root / 'freeze.json'
        self.frozen = freeze_evaluation(self.models, self.manifest, self.config, self.calibrations, self.freeze_path)
        self.prediction_root = self.root / 'locked-holdout'
        class SyntheticRunner:
            def predict_logits(self, records, batch_size=16, orientation='ab'):
                margins = np.array([(1 if r['label'] else -1) * (2 if int(r['id'].split(':')[1]) % 5 else -2) for r in records])
                return np.column_stack([np.zeros(len(margins)), margins])

        self.summary = {'schema_version': 1, 'status': 'completed', 'final_outcomes_used_for_selection': False,
                        'freeze_sha256': self.frozen['freeze_sha256'], 'manifest_sha256': self.manifest['manifest_sha256'],
                        'counts': self.manifest['counts'], 'calibrations': self.calibrations,
                        'selection': self.selection, 'selected_transformer': 'ce', 'models': {}}
        for name in MODEL_NAMES:
            self.summary['models'][name] = evaluate_model(SyntheticRunner(), self.repo, self.manifest, 'locked_holdout',
                                                         self.prediction_root / name, self.calibrations[name], self.models[name], name,
                                                         freeze_path=self.freeze_path)
        self.summary_path = self.root / 'summary.json'
        self.write_summary()

    def tearDown(self):
        self.temp.cleanup()

    def write_summary(self):
        self.summary_path.write_text(json.dumps(self.summary), encoding='utf-8')

    def run_report(self, name='uncertainty.json'):
        return bootstrap_results(self.manifest_path, self.freeze_path, self.summary_path, self.prediction_root, self.root / name)

    def update_rows(self, rows):
        path = self.prediction_root / 'ce' / 'predictions.jsonl'
        path.write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')
        self.summary['models']['ce']['predictions_sha256'] = file_sha256(path)
        (self.prediction_root / 'ce' / 'metrics.json').write_text(json.dumps(self.summary['models']['ce']), encoding='utf-8')
        self.write_summary()

    def test_complete_frozen_evidence_deterministic_and_abstention_not_filter(self):
        first = self.run_report()
        second = self.run_report('second.json')
        self.assertEqual(first, second)
        self.assertEqual(first['protocol']['draws'], 1000)
        self.assertEqual(first['evaluated_pair_count'], self.manifest['counts']['locked_holdout'])
        self.assertEqual(first['document_component_count'], self.manifest['component_counts']['locked_holdout'])
        self.assertEqual(self.summary['models']['ce']['abstention']['accepted_count'], 0)
        self.assertEqual(first['models']['ce']['accuracy']['point'], self.summary['models']['ce']['accuracy'])
        self.assertFalse(first['model_inference_performed'])
        self.assertFalse(first['configuration_reselected'])
        self.assertFalse(first['training_seed_uncertainty_estimated'])
        with self.assertRaises(FileExistsError):
            self.run_report()

    def test_partial_ids_rejected_even_after_hash_updated(self):
        rows = [json.loads(line) for line in (self.prediction_root / 'ce' / 'predictions.jsonl').read_text().splitlines()]
        self.update_rows(rows[:-1])
        with self.assertRaisesRegex(ValueError, 'cover every final pair'):
            self.run_report()

    def test_alignment_and_threshold_tamper_rejected(self):
        rows = [json.loads(line) for line in (self.prediction_root / 'ce' / 'predictions.jsonl').read_text().splitlines()]
        rows[0]['group_id'] = 'wrong-group'
        self.update_rows(rows)
        with self.assertRaisesRegex(ValueError, 'alignment'):
            self.run_report()
        rows[0]['group_id'] = next(r['group_id'] for r in self.manifest['records'] if r['id'] == rows[0]['id'])
        rows[0]['prediction'] = 1 - rows[0]['prediction']
        self.update_rows(rows)
        with self.assertRaisesRegex(ValueError, 'frozen threshold'):
            self.run_report()

    def test_declared_prediction_hash_tamper_rejected(self):
        self.summary['models']['ce']['predictions_sha256'] = '0' * 64
        (self.prediction_root / 'ce' / 'metrics.json').write_text(json.dumps(self.summary['models']['ce']), encoding='utf-8')
        self.write_summary()
        with self.assertRaisesRegex(ValueError, 'prediction SHA256'):
            self.run_report()

    def test_final_reselection_or_changed_protocol_rejected(self):
        self.summary['final_outcomes_used_for_selection'] = True
        self.write_summary()
        with self.assertRaisesRegex(ValueError, 'without final-outcome reselection'):
            self.run_report()
        self.summary['final_outcomes_used_for_selection'] = False
        self.write_summary()
        changed = copy.deepcopy(self.frozen)
        changed['config']['experiment_plan']['uncertainty']['draws'] = 500
        changed['config_sha256'] = canonical_sha256(changed['config'])
        changed['freeze_sha256'] = canonical_sha256({k: v for k, v in changed.items() if k != 'freeze_sha256'})
        self.freeze_path.write_text(json.dumps(changed), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'predeclared'):
            self.run_report()


if __name__ == '__main__':
    unittest.main()
