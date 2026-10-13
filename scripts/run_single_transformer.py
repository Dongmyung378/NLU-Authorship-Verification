"""Run exactly one matched loss on exactly one parent-assigned Kaggle GPU."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--method', choices=['ce', 'custom_asl'], required=True)
    parser.add_argument('--assigned-cuda-token', required=True)
    parser.add_argument('--assigned-visible-index', type=int, required=True)
    parser.add_argument('--pretrained-dir', type=Path)
    parser.add_argument('--epoch-coordinator-dir', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat()
    sys.path.insert(0, str(args.repo_root.resolve()))
    gpu = None
    try:
        if os.environ.get('CUDA_VISIBLE_DEVICES') != args.assigned_cuda_token or ',' in args.assigned_cuda_token:
            raise ValueError('Worker must expose exactly the GPU token assigned by its parent')
        from nlu_authorship.training import require_kaggle_gpu, run_transformer_experiment
        require_kaggle_gpu()
        import torch
        if torch.cuda.device_count() != 1:
            raise RuntimeError('A worker must see exactly one already assigned GPU')
        gpu = {'assigned_parent_visible_index': args.assigned_visible_index,
               'assigned_cuda_token': args.assigned_cuda_token, 'worker_local_index': 0,
               'visible_gpu_count': 1, 'name': torch.cuda.get_device_name(0),
               'total_bytes': torch.cuda.get_device_properties(0).total_memory}
        print(json.dumps({'stage': 'matched-worker-start', 'method': args.method, 'gpu': gpu,
                          'started_at_utc': started_utc}), flush=True)
        kwargs = {'method': args.method}
        if args.epoch_coordinator_dir is not None:
            kwargs['epoch_coordinator_dir'] = args.epoch_coordinator_dir
        report = run_transformer_experiment(args.repo_root, args.manifest, args.output,
                                            args.plan, args.pretrained_dir, **kwargs)
        if report.get('status') != 'completed' or set(report.get('methods', {})) != {args.method}:
            raise RuntimeError('Worker did not complete exactly its declared loss')
        if report.get('calibration_evaluated') is not False or report.get('locked_holdout_evaluated') is not False:
            raise RuntimeError('Training worker cannot evaluate calibration or final held-out data')
        value = {'schema_version': 1, 'status': 'completed', 'method': args.method,
                 'gpu_identity': gpu, 'started_at_utc': started_utc,
                 'completed_at_utc': datetime.now(timezone.utc).isoformat(),
                 'worker_elapsed_seconds': time.perf_counter() - started,
                 'report_filename': 'training-report.json', 'calibration_evaluated': False,
                 'locked_holdout_evaluated': False, 'execution_scope': 'actual Kaggle assigned GPU worker'}
        write_json(args.output / 'worker-completion.json', value)
        return 0
    except BaseException as error:
        value = {'schema_version': 1, 'status': 'failed', 'method': args.method,
                 'gpu_identity': gpu, 'started_at_utc': started_utc,
                 'failed_at_utc': datetime.now(timezone.utc).isoformat(),
                 'worker_elapsed_seconds': time.perf_counter() - started,
                 'error_type': type(error).__name__, 'error': str(error),
                 'partial_outputs_preserved': True, 'release_performed': False}
        write_json(args.output / 'worker-failure.json', value)
        raise


if __name__ == '__main__':
    raise SystemExit(main())
