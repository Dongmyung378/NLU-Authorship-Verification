"""Select on selection data, calibrate, freeze, then evaluate every fixed model."""
from __future__ import annotations

from datetime import datetime, timezone
import copy
import gc
import json
import math
from pathlib import Path
import time

from .calibration import apply_calibration, select_model_policy
from .data import canonical_sha256, load_records, read_manifest
from .evaluation import (evaluate_model, fit_model_calibration, freeze_evaluation,
                         predict_records_logits)
from .metrics import binary_metrics, checked_labels


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def runner_for(name, path, device='cuda'):
    if name == 'stylometric':
        from .stylometric import load_stylometric
        return load_stylometric(path)
    from .models import load_transformer
    return load_transformer(path, device=device)


def select_calibrated_pipeline(comparisons, calibrations, raw_selection, *,
                               split='selection', minimum_improvement=.002):
    """Choose the deployed loss pipeline without refitting or new inference.

    Input order is fixed by the earlier raw selection-only decision. Each
    temperature and threshold is fitted independently on calibration data;
    this final choice uses the already stored selection logits only.
    """
    if split != 'selection' or not math.isfinite(minimum_improvement) or minimum_improvement < .002:
        raise ValueError('Calibrated pipeline selection requires selection data and a>=0.002 improvement gate')
    names = {'ce', 'custom_asl', 'stylometric'}
    if set(comparisons) != names or set(calibrations) != names or set(raw_selection.get('outcomes', {})) != names:
        raise ValueError('All three fixed model pipelines need their own selection logits and calibration')
    if raw_selection.get('selection_split') != split or raw_selection.get('minimum_improvement') != minimum_improvement:
        raise ValueError('Raw orientation selection scope or predeclared improvement gate differs')
    reference_ids = comparisons['ce']['ids']
    reference_labels = checked_labels(comparisons['ce']['labels']).tolist()
    if len(reference_ids) != len(reference_labels) or len(set(reference_ids)) != len(reference_ids):
        raise ValueError('Selection IDs must uniquely match every label')
    outcomes = {}
    for name, item in comparisons.items():
        if item['ids'] != reference_ids or checked_labels(item['labels']).tolist() != reference_labels:
            raise ValueError('Calibrated selection comparisons must use the same ordered IDs and labels')
        fitted = calibrations[name]
        method = raw_selection['outcomes'][name]['selected_method']
        if fitted.get('inference_method') != method:
            raise ValueError('Calibration cannot change the earlier fixed input-order policy')
        probabilities = apply_calibration(item['logits_ab'], fitted, item['logits_ba'])
        outcomes[name] = {'metrics': binary_metrics(reference_labels, probabilities, fitted['threshold']),
                          'inference_method': method, 'temperature': fitted['temperature'],
                          'threshold': fitted['threshold'], 'calibration_parameters_sha256': canonical_sha256(fitted),
                          'parameters_fitted_on': 'calibration'}
    gain = outcomes['custom_asl']['metrics']['macro_f1'] - outcomes['ce']['metrics']['macro_f1']
    passed = gain >= minimum_improvement
    selected_name = 'custom_asl' if passed else 'ce'
    decision = {'schema_version': 1, 'decision_data': split, 'count': len(reference_labels),
                'metric': 'macro_f1', 'minimum_improvement': minimum_improvement,
                'baseline_model': 'ce', 'alternative_model': 'custom_asl',
                'alternative_macro_f1_gain': gain, 'alternative_adoption_gate_passed': passed,
                'outcomes': outcomes, 'selected_model': selected_name,
                'selection_uses_fixed_calibration_parameters': True,
                'calibration_parameters_refitted_on_selection': False,
                'final_outcomes_used': False,
                'scope': 'Complete pipelines compared on selection using independently calibration-fitted parameters and previously fixed orientations; gains cannot be attributed solely to the loss.'}
    result = copy.deepcopy(raw_selection)
    result.update({'raw_selection': copy.deepcopy(raw_selection),
                   'raw_selected_model': raw_selection['selected_model'],
                   'raw_selected_method': raw_selection['selected_method'],
                   'raw_decision_threshold': raw_selection['decision_threshold'],
                   'raw_policy_alternative_macro_f1_gain': raw_selection['alternative_macro_f1_gain'],
                   'calibrated_pipeline_selection': decision,
                   'selection_decision_basis': 'calibration_fixed_pipeline_macro_f1_on_selection',
                   'selected_model': selected_name,
                   'selected_method': outcomes[selected_name]['inference_method'],
                   'decision_threshold': outcomes[selected_name]['threshold'],
                   'alternative_macro_f1_gain': gain,
                   'loss_comparison_scope': 'Raw AB loss analysis remains in raw_selection at the common0.5 threshold; the final choice compares separately calibrated pipelines on selection and does not isolate a loss-only effect.'})
    return result


def release_models(repo_root, manifest_path, models, plan_path, output_dir, *, device='cuda', batch_size=16):
    """Final labels are accessed for prediction only after the immutable freeze."""
    start = time.perf_counter()
    repo_root, output_dir = Path(repo_root), Path(output_dir)
    if output_dir.exists():
        raise FileExistsError('Release evidence is immutable; choose a new directory')
    output_dir.mkdir(parents=True)
    plan = json.loads(Path(plan_path).read_text(encoding='utf-8'))
    if set(models) != {'ce', 'custom_asl', 'stylometric'}:
        raise ValueError('The release requires matched CE, custom ASL and the train-only stylometric model')
    manifest = read_manifest(manifest_path)
    selection = load_records(repo_root, manifest, 'selection')
    comparisons = {}
    for name, path in models.items():
        runner = runner_for(name, path, device)
        ab = predict_records_logits(runner, selection, orientation='ab', batch_size=batch_size)
        ba = predict_records_logits(runner, selection, orientation='ba', batch_size=batch_size)
        comparisons[name] = {'ids': [r['id'] for r in selection], 'labels': [r['label'] for r in selection],
                             'logits_ab': ab, 'logits_ba': ba}
        with (output_dir / ('selection-' + name + '.jsonl')).open('x', encoding='utf-8', newline='\n') as stream:
            for row, logits_ab, logits_ba in zip(selection, ab, ba):
                stream.write(json.dumps({'id': row['id'], 'label': row['label'],
                                         'logits_ab': logits_ab.tolist(), 'logits_ba': logits_ba.tolist()}, allow_nan=False) + '\n')
        del runner
        gc.collect()
        if device.startswith('cuda'):
            import torch
            torch.cuda.empty_cache()
    chosen = select_model_policy(comparisons, minimum_improvement=plan['selection']['asymmetric_loss_min_macro_f1_gain'])
    calibrations = {}
    for name, path in models.items():
        runner = runner_for(name, path, device)
        method = chosen['outcomes'][name]['selected_method']
        calibrations[name] = fit_model_calibration(
            runner, repo_root, manifest, path, method=method, batch_size=batch_size,
            minimum_temperature=plan['calibration']['temperature_bounds'][0],
            maximum_temperature=plan['calibration']['temperature_bounds'][1],
            target_coverage=plan['calibration']['abstention_target_minimum_coverage'],
            target_accuracy=plan['calibration']['abstention_target_minimum_accepted_accuracy'])
        write_json(output_dir / 'calibration' / (name + '.json'), calibrations[name])
        del runner
        gc.collect()
        if device.startswith('cuda'):
            import torch
            torch.cuda.empty_cache()
    chosen = select_calibrated_pipeline(comparisons, calibrations, chosen,
                                        minimum_improvement=plan['selection']['asymmetric_loss_min_macro_f1_gain'])
    write_json(output_dir / 'selection.json', chosen)
    configuration = {'experiment_plan': plan, 'selection': chosen,
                     'selected_transformer': chosen['selected_model'],
                     'public_default_model': 'stylometric',
                     'model_kinds': {'ce': 'transformer', 'custom_asl': 'transformer', 'stylometric': 'stylometric'},
                     'new_training_is_single_seed': True}
    freeze_path = output_dir / 'evaluation-freeze.json'
    frozen = freeze_evaluation(models, manifest, configuration, calibrations, freeze_path, source_root=repo_root)
    write_json(output_dir / 'before-final-evaluation.json',
               {'frozen_utc': datetime.now(timezone.utc).isoformat(),
                'freeze_sha256': frozen['freeze_sha256'], 'final_inference_started': False})
    results = {}
    for name, path in models.items():
        runner = runner_for(name, path, device)
        results[name] = evaluate_model(runner, repo_root, manifest, 'locked_holdout',
                                      output_dir / 'locked-holdout' / name, calibrations[name], path, name,
                                      freeze_path=freeze_path, batch_size=batch_size, source_root=repo_root)
        del runner
        gc.collect()
        if device.startswith('cuda'):
            import torch
            torch.cuda.empty_cache()
    summary = {'schema_version': 1, 'status': 'completed',
               'completed_utc': datetime.now(timezone.utc).isoformat(),
               'manifest_sha256': manifest['manifest_sha256'], 'counts': manifest['counts'],
               'freeze_sha256': frozen['freeze_sha256'], 'selected_transformer': chosen['selected_model'],
               'selection': chosen, 'calibrations': calibrations, 'models': results,
               'elapsed_seconds': time.perf_counter() - start,
               'inference_device': device, 'batch_size': batch_size,
               'final_outcomes_used_for_selection': False,
               'historical_scores_are_separate': True,
               'new_authors_or_domains_generalization_verified': False,
               'scope': 'Single-seed fresh models on document-disjoint splits; calibration fitted separately; all fixed models frozen before final inference.'}
    write_json(output_dir / 'release-summary.json', summary)
    return summary
