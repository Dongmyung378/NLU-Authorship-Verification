"""Compare a complete saved evaluation with the released frozen predictions.

This standard-library helper loads reports only. It performs no model inference,
training, calibration fitting or download. Comparison tolerances are fixed.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import sys

EXPECTED_COUNT = 4708
PROBABILITY_ATOL = 1e-5
METRIC_ATOL = 1e-6
MODEL_NAMES = ('ce', 'custom_asl', 'stylometric')
CLASS_NAMES = ['Different authors', 'Same author']
EXACT_ROW_FIELDS = ('id', 'label', 'group_id', 'split', 'prediction', 'accepted',
                    'inference_method', 'minimum_cleaned_words', 'length_bin')
PROBABILITY_FIELDS = ('raw_probability_ab', 'raw_probability_ba',
                      'raw_probability_symmetric', 'raw_probability_selected',
                      'calibrated_probability',
                      'calibrated_probability_swapped_selected_policy', 'confidence')
EXACT_NUMERIC_KEYS = {'count', 'correct', 'accuracy', 'macro_f1', 'same_author_f1',
                      'class_f1', 'class_recall', 'class_support', 'threshold',
                      'confidence_threshold', 'temperature', 'coverage',
                      'accepted_accuracy', 'prediction_mismatch_fraction'}


def canonical_sha256(value):
    encoded = json.dumps(value, sort_keys=True, separators=(',', ':'),
                         ensure_ascii=False, allow_nan=False).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def regular(path):
    path = Path(path).absolute()
    if any(p.is_symlink() or getattr(p, 'is_junction', lambda: False)()
           for p in (path, *path.parents)) or not path.is_file():
        raise ValueError('Expected a regular file without symbolic parents')
    return path


def file_sha256(path):
    digest = hashlib.sha256()
    with regular(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON object key')
        result[key] = value
    return result


def _reject_nonfinite(value):
    raise ValueError('Nonfinite JSON number: ' + value)


def decode_json(raw):
    return json.loads(raw, object_pairs_hook=_unique_object,
                      parse_constant=_reject_nonfinite)


def read_json(path):
    return decode_json(regular(path).read_text(encoding='utf-8'))


def source_inventory(root):
    paths = set((root / 'nlu_authorship').rglob('*.py'))
    paths.update((root / 'scripts').rglob('*.py'))
    paths.update((root / 'scripts').rglob('*.ps1'))
    paths.update((root / 'configs').rglob('*.json'))
    paths.update(root.glob('requirements*.txt'))
    paths.update(root.glob('launch-nlu.*'))
    paths.update(p for p in (root / 'run.ps1', root / 'pyproject.toml') if p.is_file())
    return {p.relative_to(root).as_posix(): file_sha256(p)
            for p in sorted(paths) if p.is_file()}


def release_context(root, model_name):
    manifest = read_json(root / 'reports/split-manifest.json')
    freeze = read_json(root / 'reports/evaluation-freeze.json')
    summary = read_json(root / 'reports/release-summary.json')
    if manifest.get('manifest_sha256') != canonical_sha256(
            {k: v for k, v in manifest.items() if k != 'manifest_sha256'}):
        raise ValueError('Split manifest checksum mismatch')
    if freeze.get('freeze_sha256') != canonical_sha256(
            {k: v for k, v in freeze.items() if k != 'freeze_sha256'}):
        raise ValueError('Evaluation freeze checksum mismatch')
    if (freeze.get('manifest_sha256') != manifest['manifest_sha256'] or
            freeze.get('full_manifest_sha256') != canonical_sha256(manifest) or
            freeze.get('source_csvs') != manifest['source_csvs']):
        raise ValueError('Frozen manifest or dataset binding differs')
    if (freeze.get('config_sha256') != canonical_sha256(freeze['config']) or
            freeze.get('calibrations_sha256') != canonical_sha256(freeze['calibrations'])):
        raise ValueError('Frozen configuration or calibration checksum differs')
    if freeze.get('source_sha256') != source_inventory(root):
        raise ValueError('Executable source/configuration/dependencies changed after freeze')
    if freeze.get('class_ids') != [0, 1] or freeze.get('class_names') != CLASS_NAMES:
        raise ValueError('Frozen binary class mapping differs')
    if summary.get('status') != 'completed' or summary.get('freeze_sha256') != freeze['freeze_sha256']:
        raise ValueError('Reference release is incomplete or binds a different freeze')
    records = [r for r in manifest['records'] if r['split'] == 'locked_holdout']
    if len(records) != EXPECTED_COUNT or len({r['id'] for r in records}) != EXPECTED_COUNT:
        raise ValueError('Locked holdout must contain exactly4708 unique ordered records')
    minimal = [{k: r[k] for k in ('id', 'label', 'text_1_sha256', 'text_2_sha256', 'group_id')}
               for r in records]
    calibration = freeze['calibrations'][model_name]
    bindings = {
        'schema_version': 1, 'model_name': model_name,
        'model_content_sha256': freeze['models'][model_name]['content_sha256'],
        'model_artifacts': freeze['models'][model_name]['files'],
        'split': 'locked_holdout', 'manifest_sha256': manifest['manifest_sha256'],
        'source_csvs': manifest['source_csvs'],
        'evaluation_records_sha256': canonical_sha256(minimal),
        'freeze_sha256': freeze['freeze_sha256'],
        'source_sha256': freeze['source_sha256'],
        'calibration_sha256': canonical_sha256(calibration),
        'inference_method': calibration['inference_method'],
        'class_names': freeze['class_names'], 'threshold': calibration['threshold'],
        'historical_exposure': manifest['historical_exposure'],
        'author_or_domain_separation_verified': False,
    }
    return records, bindings, summary['models'][model_name]


def _metric_differences(first, second, path=''):
    differences = []
    if type(first) is not type(second):
        return [{'field': path, 'reason': 'value types differ'}]
    if isinstance(first, dict):
        if first.keys() != second.keys():
            differences.append({'field': path, 'reason': 'object keys differ'})
        for key in sorted(first.keys() & second.keys()):
            if key != 'predictions_sha256':
                differences += _metric_differences(first[key], second[key], path + '.' + key)
    elif isinstance(first, list):
        if len(first) != len(second):
            return [{'field': path, 'reason': 'list lengths differ'}]
        for i, (a, b) in enumerate(zip(first, second)):
            differences += _metric_differences(a, b, path + '.' + str(i))
    elif isinstance(first, float):
        if not math.isfinite(first) or not math.isfinite(second):
            return [{'field': path, 'reason': 'nonfinite number'}]
        exact = any(part in EXACT_NUMERIC_KEYS for part in path.split('.'))
        tolerance = 0.0 if exact else METRIC_ATOL
        if abs(first - second) > tolerance:
            differences.append({'field': path, 'evaluated': first, 'reference': second,
                                'absolute_tolerance': tolerance})
    elif first != second:
        differences.append({'field': path, 'evaluated': first, 'reference': second})
    return differences


def _prediction_rows(folder, metrics, expected_records):
    path = regular(folder / 'predictions.jsonl')
    digest = file_sha256(path)
    if digest != metrics.get('predictions_sha256'):
        raise ValueError('Prediction file SHA differs from its metrics record')
    rows = [decode_json(line) for line in path.read_text(encoding='utf-8').splitlines()]
    if len(rows) != EXPECTED_COUNT or len({r['id'] for r in rows}) != EXPECTED_COUNT:
        raise ValueError('Prediction file must contain exactly4708 unique IDs')
    if [r['id'] for r in rows] != [r['id'] for r in expected_records]:
        raise ValueError('Prediction IDs differ from exact manifest order')
    for row, record in zip(rows, expected_records):
        for field in ('id', 'label', 'group_id'):
            if row.get(field) != record[field]:
                raise ValueError('Prediction record differs from manifest: ' + field)
        if type(row.get('prediction')) is not int or row['prediction'] not in (0, 1):
            raise ValueError('Invalid discrete class prediction')
        if type(row.get('accepted')) is not bool or type(row.get('label')) is not int:
            raise ValueError('Invalid discrete acceptance or label type')
        for field in PROBABILITY_FIELDS:
            value = row[field]
            if type(value) not in (float, int) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError('Invalid stored prediction probability')
    return rows, digest


def compare_reports(evaluation_dir, reference_dir, records, bindings, reference_summary=None):
    evaluated, reference = read_json(evaluation_dir / 'metrics.json'), read_json(reference_dir / 'metrics.json')
    for role, metrics in (('evaluated', evaluated), ('reference', reference)):
        for key, expected in bindings.items():
            if metrics.get(key) != expected:
                raise ValueError(role + ' report differs from frozen binding: ' + key)
    if reference_summary is not None and reference != reference_summary:
        raise ValueError('Reference metrics differ from the completed release summary')
    left, left_sha = _prediction_rows(evaluation_dir, evaluated, records)
    right, right_sha = _prediction_rows(reference_dir, reference, records)
    metric_differences = _metric_differences(evaluated, reference)
    largest = {field: 0.0 for field in PROBABILITY_FIELDS}
    mismatches = Counter({field: 0 for field in EXACT_ROW_FIELDS})
    examples = []
    for first, second in zip(left, right):
        if first.keys() != second.keys():
            raise ValueError('Prediction record keys differ')
        for field in EXACT_ROW_FIELDS:
            if first[field] != second[field] or type(first[field]) is not type(second[field]):
                mismatches[field] += 1
                if len(examples) < 20:
                    examples.append({'id': first['id'], 'field': field,
                                     'evaluated': first[field], 'reference': second[field]})
        for field in PROBABILITY_FIELDS:
            largest[field] = max(largest[field], abs(first[field] - second[field]))
    for folder, metrics in ((evaluation_dir, evaluated), (reference_dir, reference)):
        confusion = read_json(folder / 'confusion.json')
        if (confusion.get('matrix') != metrics['confusion'] or confusion.get('class_ids') != [0, 1] or
                confusion.get('class_names') != bindings['class_names'] or
                confusion.get('rows') != 'true label' or confusion.get('columns') != 'predicted label'):
            raise ValueError('Confusion report differs from class/metric bindings')
    passed = not metric_differences and not any(mismatches.values()) and max(largest.values()) <= PROBABILITY_ATOL
    return {
        'status': 'passed' if passed else 'failed', 'count': len(left),
        'all_frozen_bindings_match': True, 'record_ids_match_manifest_exactly': True,
        'exact_row_mismatch_counts': dict(mismatches), 'row_mismatch_examples': examples,
        'metric_difference_count': len(metric_differences),
        'metric_differences': metric_differences[:20],
        'probability_absolute_tolerance': PROBABILITY_ATOL,
        'numeric_metric_absolute_tolerance': METRIC_ATOL,
        'maximum_absolute_probability_difference': largest,
        'evaluated_predictions_sha256': left_sha, 'reference_predictions_sha256': right_sha,
        'prediction_files_byte_identical': left_sha == right_sha,
        'exact_discrete_predictions_acceptance_counts_f1_accuracy_required': True,
        'no_inference_refit_reselection_or_tolerance_adjustment': True,
        'anonymous_public_model_download_verified': False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-name', required=True, choices=MODEL_NAMES)
    parser.add_argument('--evaluation-dir', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path, help='New comparison JSON file; never overwritten')
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    output = args.output.absolute()
    if output.exists() or any(p.is_symlink() or getattr(p, 'is_junction', lambda: False)()
                              for p in (output, *output.parents)):
        parser.error('Comparison output must be new and have no symbolic parents')
    if output.is_relative_to(root / 'reports'):
        parser.error('Use an output outside the released reports directory')
    result = {'schema_version': 1, 'model_name': args.model_name,
              'created_utc': datetime.now(timezone.utc).isoformat(),
              'comparison_host_os': platform.system(),
              'reference_execution_device': 'Kaggle-hosted CPU stylometric ensemble' if args.model_name == 'stylometric' else 'Kaggle CUDA/T4 RoBERTa-Large',
              'evaluation_execution_device': 'Not inferred from the comparison host; see the original evaluation environment receipt.',
              'helper_sha256': file_sha256(Path(__file__)),
              'scope': 'Saved full-evaluation report comparison only; not an inference run or public-download check.'}
    try:
        records, bindings, summary = release_context(root, args.model_name)
        result.update({'freeze_sha256': bindings['freeze_sha256'],
                       'manifest_sha256': bindings['manifest_sha256'],
                       'frozen_source_file_count': len(bindings['source_sha256']),
                       'evaluated_metrics_file_sha256': file_sha256(args.evaluation_dir / 'metrics.json'),
                       'reference_metrics_file_sha256': file_sha256(root / 'reports/locked-holdout' / args.model_name / 'metrics.json')})
        result.update(compare_reports(args.evaluation_dir, root / 'reports/locked-holdout' / args.model_name,
                                      records, bindings, summary))
    except (ValueError, KeyError, TypeError, OSError) as error:
        result.update({'status': 'failed', 'failure_reason': str(error),
                       'no_inference_refit_reselection_or_tolerance_adjustment': True,
                       'anonymous_public_model_download_verified': False})
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status': result['status'], 'model_name': args.model_name,
                      'count': result.get('count'), 'output': str(output)}))
    return 0 if result['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
