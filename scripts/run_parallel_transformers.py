"""Two owned Kaggle workers, shared stopping and atomic matched-result promotion."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

METHODS = ('ce', 'custom_asl')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def assigned_devices():
    from nlu_authorship.training import require_kaggle_gpu
    require_kaggle_gpu()
    import torch
    count = torch.cuda.device_count()
    if count != 2:
        raise RuntimeError('Parallel matched training requires exactly two GPUs assigned to this Kaggle kernel')
    mask = os.environ.get('CUDA_VISIBLE_DEVICES')
    tokens = [str(i) for i in range(count)] if mask is None else [v.strip() for v in mask.split(',')]
    if len(tokens) != count or len(set(tokens)) != count or any(not token or token == '-1' for token in tokens):
        raise ValueError('Assigned CUDA mask is inconsistent; never guess other GPU indices')
    devices = [{'visible_index': i, 'token': tokens[i], 'name': torch.cuda.get_device_name(i),
                'total_bytes': torch.cuda.get_device_properties(i).total_memory} for i in range(count)]
    if any('T4' not in device['name'] for device in devices):
        raise RuntimeError('The declared parallel profile requires two actually assigned T4 devices')
    return devices


def _stop_owned(processes, grace_seconds):
    terminated = []
    for method, item in processes.items():
        process = item['process']
        if process.poll() is None:
            try:
                process.terminate()
            except OSError:
                if process.poll() is None:
                    raise
            terminated.append({'method': method, 'pid': process.pid})
    deadline = time.perf_counter() + grace_seconds
    for item in processes.values():
        process = item['process']
        if process.poll() is None:
            try:
                process.wait(timeout=max(.01, deadline - time.perf_counter()))
            except subprocess.TimeoutExpired:
                process.kill()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    terminated.append({'pid': process.pid, 'unresponsive_after_kill': True})
    return terminated


def _validate_worker(method, directory, completion, plan, manifest_hash, plan_hash, source, schedules, require_gpu):
    if completion.get('status') != 'completed' or completion.get('method') != method:
        raise ValueError('Worker completion identity/status differs')
    if completion.get('calibration_evaluated') is not False or completion.get('locked_holdout_evaluated') is not False:
        raise ValueError('Worker performed forbidden evaluation')
    if completion.get('report_filename') != 'training-report.json':
        raise ValueError('Worker report path must be the declared relative filename')
    elapsed = float(completion.get('worker_elapsed_seconds', -1))
    if not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError('Worker elapsed duration must be finite and nonnegative')
    for key in ('started_at_utc', 'completed_at_utc'):
        if datetime.fromisoformat(completion[key]).tzinfo is None:
            raise ValueError('Observed worker timestamps must include their timezone')
    if require_gpu and completion.get('gpu_identity', {}).get('visible_gpu_count') != 1:
        raise ValueError('Worker did not isolate one assigned GPU')
    report = read_json(directory / 'training-report.json')
    if report.get('status') != 'completed' or set(report.get('methods', {})) != {method}:
        raise ValueError('Each worker must complete exactly one declared method')
    bindings = {'plan': plan, 'manifest_file_sha256': manifest_hash, 'plan_sha256': plan_hash,
                'source_snapshot': source, 'epoch_schedule_sha256': schedules,
                'calibration_evaluated': False, 'locked_holdout_evaluated': False, 'seed_count': 1}
    if any(report.get(key) != expected for key, expected in bindings.items()):
        raise ValueError('Worker plan, schedule, source or data binding differs')
    initial = report.get('initial_state_sha256')
    checkpoint = report['methods'][method]
    if not isinstance(initial, str) or len(initial) != 64 or checkpoint.get('initial_state_sha256') != initial:
        raise ValueError('Worker initial-state fingerprint is absent or inconsistent')
    epochs = checkpoint.get('epochs', [])
    if not epochs or len(epochs) > plan['epochs']:
        raise ValueError('Worker did not complete a valid matched epoch count')
    if [item.get('epoch') for item in epochs] != list(range(1, len(epochs) + 1)):
        raise ValueError('Epoch sequence is incomplete')
    if any(item.get('schedule_sha256') != schedules[i] for i, item in enumerate(epochs)):
        raise ValueError('Actual epoch schedules differ from the declared schedule')
    from nlu_authorship.models import verify_artifact_inventory
    if (directory / method).is_symlink():
        raise ValueError('Worker model folder cannot be a symlink')
    verify_artifact_inventory(directory / method, ('model.safetensors', 'config.json', 'training-config.json', 'checkpoint-report.json'))
    weight = directory / method / 'model.safetensors'
    if digest(weight) != checkpoint.get('model_sha256') or weight.stat().st_size != checkpoint.get('model_bytes'):
        raise ValueError('Saved checkpoint differs from its worker report')
    if read_json(directory / method / 'checkpoint-report.json') != checkpoint:
        raise ValueError('Saved checkpoint metadata differs from the paired report')
    return report


def _coordinate_ready_epochs(coordinator, epoch, plan, source, manifest_hash, plan_hash, best, idle, patience, decisions):
    if decisions and decisions[-1]['action'] == 'stop':
        return epoch, idle
    paths = {method: coordinator / f'epoch-{epoch:04d}-{method}.json' for method in METHODS}
    if not all(path.is_file() for path in paths.values()):
        return epoch, idle
    improved = {}
    for method, path in paths.items():
        row = read_json(path)
        if row.get('method') != method or row.get('epoch') != epoch or row.get('manifest_file_sha256') != manifest_hash or row.get('plan_sha256') != plan_hash:
            raise ValueError('Shared epoch barrier identity or data/plan binding differs')
        item = row['item']
        metric = float(item['selection_macro_f1_ab_threshold_0_5'])
        if not math.isfinite(metric) or not 0 <= metric <= 1 or item.get('epoch') != epoch:
            raise ValueError('Shared stopping uses only finite raw AB selection macro F1')
        improved[method] = metric > best[method]
        best[method] = max(best[method], metric)
    states = [read_json(paths[method]).get('initial_state_sha256') for method in METHODS]
    if states[0] != states[1] or not isinstance(states[0], str) or len(states[0]) != 64:
        raise ValueError('Worker initializations differ at the shared epoch barrier')
    microbatches = [read_json(paths[method]).get('microbatch') for method in METHODS]
    if (microbatches[0] != microbatches[1] or isinstance(microbatches[0], bool)
            or not isinstance(microbatches[0], int) or microbatches[0] < 1):
        raise ValueError('Matched worker microbatch settings differ at the shared epoch barrier')
    idle = 0 if any(improved.values()) else idle + 1
    stop = idle >= patience or epoch >= plan['epochs']
    decision = {'schema_version': 1, 'epoch': epoch, 'action': 'stop' if stop else 'continue',
                'reason': 'common_selection_patience' if idle >= patience else 'matched_epoch_cap' if stop else 'selection_improved_or_patience_remaining',
                'best_raw_ab_selection_macro_f1': dict(best), 'improved_this_epoch': improved,
                'consecutive_epochs_neither_improved': idle, 'common_patience': patience,
                'selection_only': True, 'holdout_used': False}
    write_json(coordinator / f'epoch-{epoch:04d}-decision.json', decision)
    decisions.append(decision)
    return epoch + 1, idle


def run_parallel_transformers(repo_root, manifest_path, output_dir, plan_path, pretrained_dir=None, *,
                              wall_timeout_seconds=36000, gpu_budget_seconds=72000,
                              common_patience=3, stop_grace_seconds=10, poll_seconds=.25,
                              require_gpu=True, worker_script=None, devices=None):
    """Default path is actual Kaggle; synthetic subprocess injection is unit-test only."""
    repo = Path(repo_root).resolve()
    output = Path(output_dir).resolve()
    manifest_path, plan_path = Path(manifest_path).resolve(), Path(plan_path).resolve()
    if output.exists():
        raise FileExistsError('Canonical output must be absent to permit atomic all-method promotion')
    if not 0 < wall_timeout_seconds <= 36000 or not 0 < gpu_budget_seconds <= 72000 or common_patience < 1 or stop_grace_seconds <= 0 or poll_seconds <= 0:
        raise ValueError('Parent caps cannot exceed10wall-hours/20assigned-worker-hours')
    manifest_hash, plan_hash = digest(manifest_path), digest(plan_path)
    manifest = read_json(manifest_path)
    if require_gpu:
        if worker_script is not None or devices is not None:
            raise ValueError('Actual GPU orchestration cannot override the worker or assigned-device probe')
        from nlu_authorship.training import validated_plan, _source_snapshot, schedule_sha256
        from nlu_authorship.evaluation import source_inventory
        plan = validated_plan(plan_path)
        source = _source_snapshot(repo)
        full_source = source_inventory(repo)
        train = [r for r in manifest['records'] if r['split'] == 'train']
        schedules = [schedule_sha256(train, plan['seed'], epoch) for epoch in range(plan['epochs'])]
        devices = assigned_devices()
    else:
        if worker_script is None or devices is None:
            raise ValueError('Synthetic CPU tests must explicitly provide their worker fixture and devices')
        plan = read_json(plan_path)
        source = plan['fixture_source_snapshot']
        schedules = plan['fixture_schedules']
        full_source = None
    script = Path(worker_script) if worker_script else repo / 'scripts' / 'run_single_transformer.py'
    if len(devices) != 2 or devices[0]['token'] == devices[1]['token']:
        raise ValueError('Two distinct already-assigned device tokens are required')
    output.parent.mkdir(parents=True, exist_ok=True)
    work = output.with_name(output.name + '-parallel')
    work.mkdir(exist_ok=False)
    workers = work / 'workers'
    logs = work / 'logs'
    coordinator = work / 'coordinator'
    for folder in (workers, logs, coordinator):
        folder.mkdir()
    started = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat()
    processes, completed, reports, decisions = {}, {}, {}, []
    best = {method: -math.inf for method in METHODS}
    next_epoch, idle = 1, 0
    try:
        for method, device in zip(METHODS, devices):
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = device['token']
            env['PYTHONNOUSERSITE'] = '1'
            for key in ('PYTHONHOME', 'PYTHONPATH', 'VIRTUAL_ENV'):
                env.pop(key, None)
            env.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
            directory = workers / method
            command = [sys.executable, '-I', '-X', 'utf8', str(script), '--repo-root', str(repo),
                       '--manifest', str(manifest_path), '--output', str(directory), '--plan', str(plan_path),
                       '--method', method, '--assigned-cuda-token', device['token'],
                       '--assigned-visible-index', str(device['visible_index']), '--epoch-coordinator-dir', str(coordinator)]
            if pretrained_dir is not None:
                command.extend(['--pretrained-dir', str(Path(pretrained_dir).resolve())])
            stream = (logs / f'{method}.log').open('w', encoding='utf-8')
            kwargs = {'creationflags': subprocess.CREATE_NO_WINDOW} if os.name == 'nt' else {'start_new_session': True}
            process = subprocess.Popen(command, cwd=repo, env=env, stdout=stream, stderr=subprocess.STDOUT, **kwargs)
            processes[method] = {'process': process, 'stream': stream, 'directory': directory,
                                 'launch_monotonic': time.perf_counter(), 'launched_at_utc': datetime.now(timezone.utc).isoformat(),
                                 'command': command, 'device': device}
        write_json(work / 'parallel-status.json', {'status': 'running', 'started_at_utc': started_utc,
                    'owned_pids': {method: item['process'].pid for method, item in processes.items()}, 'devices': devices})
        while len(completed) < len(METHODS):
            now = time.perf_counter()
            elapsed = now - started
            assigned_seconds = sum((item.get('exit_monotonic', now) - item['launch_monotonic']) for item in processes.values())
            if elapsed > wall_timeout_seconds or assigned_seconds > gpu_budget_seconds:
                raise TimeoutError('Predeclared parent wall/assigned-GPU budget reached; incomplete pairs are failures')
            next_epoch, idle = _coordinate_ready_epochs(coordinator, next_epoch, plan, source, manifest_hash, plan_hash, best, idle, common_patience, decisions)
            for method, item in processes.items():
                code = item['process'].poll()
                if code is None or method in completed:
                    continue
                item['exit_monotonic'] = now
                if code:
                    raise RuntimeError(f'Owned{method}worker failed with exit code{code}; no release permitted')
                path = item['directory'] / 'worker-completion.json'
                if not path.is_file():
                    raise RuntimeError('A worker exited without its complete success receipt')
                completion = read_json(path)
                report = _validate_worker(method, item['directory'], completion, plan, manifest_hash, plan_hash, source, schedules, require_gpu)
                if require_gpu and completion['gpu_identity']['assigned_cuda_token'] != item['device']['token']:
                    raise ValueError('Worker ran outside its assigned CUDA token')
                completed[method], reports[method] = completion, report
            if len(completed) < len(METHODS):
                time.sleep(poll_seconds)
        if reports['ce']['initial_state_sha256'] != reports['custom_asl']['initial_state_sha256']:
            raise ValueError('Matched CE/ASL initial-state hashes differ')
        if reports['ce']['pretrained_inventory'] != reports['custom_asl']['pretrained_inventory']:
            raise ValueError('Matched pretrained source inventories differ')
        if reports['ce']['profile']['microbatch'] != reports['custom_asl']['profile']['microbatch']:
            raise ValueError('Matched worker microbatch settings differ before promotion')
        epoch_counts = {method: len(reports[method]['methods'][method]['epochs']) for method in METHODS}
        if len(set(epoch_counts.values())) != 1:
            raise ValueError('Matched methods completed different epoch counts')
        actual_epochs = next(iter(epoch_counts.values()))
        if len(decisions) != actual_epochs or decisions[-1]['action'] != 'stop' or decisions[-1]['epoch'] != actual_epochs:
            raise ValueError('Every completed epoch must obey the common stopping barrier')
        if actual_epochs < plan['epochs'] and (not decisions or decisions[-1]['action'] != 'stop' or decisions[-1]['reason'] != 'common_selection_patience' or decisions[-1]['epoch'] != actual_epochs):
            raise ValueError('Truncated training is not a successful pair without the shared predeclared stop decision')
        if digest(manifest_path) != manifest_hash or digest(plan_path) != plan_hash:
            raise ValueError('Dataset/plan changed during parallel training')
        if require_gpu:
            if _source_snapshot(repo) != source or source_inventory(repo) != full_source:
                raise ValueError('Bound executable source changed during parallel training')
        selection = [r for r in manifest['records'] if r['split'] == 'selection']
        selection_bindings = [(r['id'], r['label']) for r in selection]
        for method in METHODS:
            path = workers / method / f'selection-{method}-logits.jsonl'
            rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
            if [(row['id'], row['label']) for row in rows] != selection_bindings:
                raise ValueError('Selection predictions do not match exact ordered selection IDs/labels')
            if any(len(row.get(key, [])) != 2 or any(not math.isfinite(float(value)) for value in row[key]) for row in rows for key in ('logits_ab', 'logits_ba')):
                raise ValueError('Selection logits are incomplete or nonfinite')
        staging = work / 'canonical-staging'
        staging.mkdir()
        for method in METHODS:
            source_model = workers / method / method
            target_model = staging / method
            if source_model.is_symlink() or not source_model.resolve().is_relative_to(work.resolve()) or not target_model.resolve().is_relative_to(work.resolve()):
                raise ValueError('Model promotion must stay inside this owned parallel workspace')
            source_model.rename(target_model)
            shutil.copyfile(workers / method / f'selection-{method}-logits.jsonl', staging / f'selection-{method}-logits.jsonl')
        elapsed = time.perf_counter() - started
        if elapsed > wall_timeout_seconds:
            raise TimeoutError('Predeclared parent deadline reached before atomic promotion')
        merged = {'schema_version': 1, 'status': 'completed', 'parallel_training': True,
                  'execution_scope': 'actual Kaggle2assignedT4' if require_gpu else 'synthetic CPU subprocess control-flow test',
                  'plan': plan, 'plan_sha256': plan_hash, 'manifest_file_sha256': manifest_hash,
                  'source_snapshot': source, 'full_source_snapshot': full_source,
                  'pretrained_inventory': reports['ce']['pretrained_inventory'],
                  'initial_state_sha256': reports['ce']['initial_state_sha256'], 'epoch_schedule_sha256': schedules,
                  'train_count': reports['ce']['train_count'], 'selection_count': reports['ce']['selection_count'],
                  'seed_count': 1, 'methods': {method: reports[method]['methods'][method] for method in METHODS},
                  'profiles': {method: reports[method]['profile'] for method in METHODS},
                  'workers': completed, 'parent_devices': devices, 'common_stop_decisions': decisions,
                  'actual_matched_epochs': actual_epochs, 'common_patience': common_patience,
                  'parent_started_at_utc': started_utc, 'parent_completed_at_utc': datetime.now(timezone.utc).isoformat(),
                  'parent_wall_seconds': elapsed, 'training_seconds': elapsed,
                  'sum_assigned_worker_seconds': sum(value['worker_elapsed_seconds'] for value in completed.values()),
                  'gpu_time_scope': 'Sum of worker wall durations while assigned GPUs are reserved; includes CPU preparation/barrier waiting, not device-kernel-only busy time.',
                  'parent_wall_budget_seconds': wall_timeout_seconds, 'assigned_gpu_budget_seconds': gpu_budget_seconds,
                  'calibration_evaluated': False, 'locked_holdout_evaluated': False,
                  'private_worker_evidence_directory': work.name, 'atomic_all_method_promotion': True}
        write_json(staging / 'training-report.json', merged)
        staging.rename(output)
        write_json(work / 'parallel-status.json', {'status': 'completed', 'canonical_output': output.name,
                                                  'parent_wall_seconds': elapsed, 'actual_matched_epochs': actual_epochs})
        return merged
    except BaseException as error:
        write_json(coordinator / 'abort.json', {'status': 'failed', 'error_type': type(error).__name__, 'error': str(error)})
        terminated = _stop_owned(processes, stop_grace_seconds)
        failure = {'schema_version': 1, 'status': 'failed', 'error_type': type(error).__name__, 'error': str(error),
                   'parent_wall_seconds': time.perf_counter() - started, 'owned_workers_terminated': terminated,
                   'owned_worker_return_codes': {method: item['process'].poll() for method, item in processes.items()},
                   'completed_methods_not_promoted': list(completed), 'partial_outputs_preserved': True,
                   'canonical_output_created': output.exists(), 'release_performed': False}
        write_json(work / 'parallel-failure.json', failure)
        write_json(work / 'parallel-status.json', failure)
        raise
    finally:
        for item in processes.values():
            item['stream'].close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--pretrained-dir', type=Path)
    parser.add_argument('--wall-timeout-seconds', type=float, default=36000)
    parser.add_argument('--gpu-budget-seconds', type=float, default=72000)
    parser.add_argument('--common-patience', type=int, default=3)
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo_root.resolve()))
    report = run_parallel_transformers(args.repo_root, args.manifest, args.output, args.plan,
        args.pretrained_dir, wall_timeout_seconds=args.wall_timeout_seconds,
        gpu_budget_seconds=args.gpu_budget_seconds, common_patience=args.common_patience)
    print(json.dumps({'status': report['status'], 'parent_wall_seconds': report['parent_wall_seconds'],
                      'actual_matched_epochs': report['actual_matched_epochs']}))


if __name__ == '__main__':
    raise SystemExit(main())
