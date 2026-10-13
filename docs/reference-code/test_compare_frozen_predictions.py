"""Meaningful report corruption checks without models or network access."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('compare_frozen_predictions', Path(__file__).with_name('compare_frozen_predictions.py'))
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


class ComparisonCorruptionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.evaluated, self.reference = self.root / 'evaluated', self.root / 'reference'
        self.evaluated.mkdir()
        self.reference.mkdir()
        self.records = [{'id': f'pair:{i:06d}', 'label': i % 2, 'group_id': f'group:{i//2}'}
                        for i in range(helper.EXPECTED_COUNT)]
        self.bindings = {'model_name': 'stylometric', 'freeze_sha256': 'frozen-checksum',
                         'source_sha256': {'module.py': 'source-checksum'},
                         'calibration_sha256': 'calibration-checksum',
                         'class_names': helper.CLASS_NAMES}
        self.rows = []
        for record in self.records:
            probability = 0.75 if record['label'] else 0.25
            self.rows.append({**record, 'split': 'locked_holdout', 'prediction': record['label'],
                              'accepted': True, 'inference_method': 'ab',
                              'minimum_cleaned_words': 10, 'length_bin': 'under_50',
                              **{field: probability for field in helper.PROBABILITY_FIELDS}})
        half = helper.EXPECTED_COUNT // 2
        self.metrics = {**self.bindings, 'count': helper.EXPECTED_COUNT,
                        'correct': helper.EXPECTED_COUNT, 'accuracy': 1.0, 'macro_f1': 1.0,
                        'confusion': [[half, 0], [0, half]], 'ece10': 0.25,
                        'abstention': {'accepted_count': helper.EXPECTED_COUNT, 'coverage': 1.0,
                                       'accepted_accuracy': 1.0}}
        self.write(self.reference)
        self.write(self.evaluated)

    def write(self, folder):
        (folder / 'predictions.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in self.rows), encoding='utf-8')
        (folder / 'metrics.json').write_text(json.dumps({**self.metrics, 'predictions_sha256': helper.file_sha256(folder / 'predictions.jsonl')}), encoding='utf-8')
        (folder / 'confusion.json').write_text(json.dumps({'class_ids': [0, 1], 'class_names': helper.CLASS_NAMES,
                                                        'rows': 'true label', 'columns': 'predicted label',
                                                        'matrix': self.metrics['confusion']}), encoding='utf-8')

    def compare(self):
        return helper.compare_reports(self.evaluated, self.reference, self.records, self.bindings)

    def test_exact_reports_pass(self):
        self.assertEqual(self.compare()['status'], 'passed')

    def test_small_probability_difference_passes_without_byte_identity(self):
        self.rows[0]['calibrated_probability'] += 5e-6
        self.write(self.evaluated)
        result = self.compare()
        self.assertEqual(result['status'], 'passed')
        self.assertFalse(result['prediction_files_byte_identical'])

    def test_label_flip_is_rejected_even_with_updated_file_sha(self):
        self.rows[0]['label'] = 1
        self.write(self.evaluated)
        with self.assertRaisesRegex(ValueError, 'manifest: label'):
            self.compare()

    def test_acceptance_flip_fails(self):
        self.rows[0]['accepted'] = False
        self.write(self.evaluated)
        self.assertEqual(self.compare()['status'], 'failed')

    def test_probability_above_fixed_tolerance_fails(self):
        self.rows[0]['calibrated_probability'] += 2e-5
        self.write(self.evaluated)
        self.assertEqual(self.compare()['status'], 'failed')

    def test_wrong_freeze_binding_is_rejected(self):
        self.metrics['freeze_sha256'] = 'different-freeze'
        self.write(self.evaluated)
        with self.assertRaisesRegex(ValueError, 'frozen binding: freeze_sha256'):
            self.compare()

    def test_misordered_predictions_are_rejected(self):
        self.rows[0], self.rows[1] = self.rows[1], self.rows[0]
        self.write(self.evaluated)
        with self.assertRaisesRegex(ValueError, 'manifest order'):
            self.compare()

    def test_prediction_sha_must_match_metrics(self):
        with (self.evaluated / 'predictions.jsonl').open('a', encoding='utf-8') as stream:
            stream.write('\n')
        with self.assertRaisesRegex(ValueError, 'file SHA'):
            self.compare()

    def test_f1_requires_exact_match(self):
        self.metrics['macro_f1'] -= 1e-12
        self.write(self.evaluated)
        self.assertEqual(self.compare()['status'], 'failed')

    def test_existing_output_is_preserved(self):
        output = self.root / 'existing.json'
        output.write_text('preserve', encoding='utf-8')
        with self.assertRaises(SystemExit):
            helper.main(['--model-name', 'stylometric', '--evaluation-dir', str(self.evaluated), '--output', str(output)])
        self.assertEqual(output.read_text(encoding='utf-8'), 'preserve')


if __name__ == '__main__':
    unittest.main()
