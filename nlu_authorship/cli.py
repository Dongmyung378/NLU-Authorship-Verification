"""Run released models and frozen evaluations without local GPU setup."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def write_result(result, output=None):
    serialized = json.dumps(result, ensure_ascii=False, indent=2) + '\n'
    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('x', encoding='utf-8', newline='\n') as stream:
            stream.write(serialized)
    print(serialized, end='')


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    d = sub.add_parser('doctor', help='Check pinned dependencies and a real CPU tensor operation')
    d.add_argument('--output', type=Path)
    dl = sub.add_parser('download', help='Download and hash-check the published models')
    dl.add_argument('--manifest', type=Path, default=ROOT / 'reports/artifact-download.json')
    dl.add_argument('--output', type=Path, default=ROOT / 'artifacts/runtime')
    dl.add_argument('--model', choices=['stylometric', 'transformer', 'ce', 'custom_asl', 'all'], default='stylometric')
    dl.add_argument('--verify-only', action='store_true')
    vd = sub.add_parser('verify-data', help='Verify CSV hashes, split membership and document isolation')
    vd.add_argument('--manifest', type=Path, default=ROOT / 'reports/split-manifest.json')
    vd.add_argument('--data-root', type=Path, default=ROOT)
    prep = sub.add_parser('prepare-split', help='Build a new document-component split without training')
    prep.add_argument('--data-root', type=Path, default=ROOT)
    prep.add_argument('--output', type=Path, required=True)
    pr = sub.add_parser('predict', help='Compare two passages using a released calibrated model')
    pr.add_argument('--model', choices=['stylometric', 'transformer'], default='stylometric')
    pr.add_argument('--model-dir', type=Path)
    pr.add_argument('--model-name')
    pr.add_argument('--freeze', type=Path, default=ROOT / 'reports/evaluation-freeze.json')
    for side in ('1', '2'):
        group = pr.add_mutually_exclusive_group()
        group.add_argument('--text-' + side)
        group.add_argument('--text-file-' + side, type=Path)
    pr.add_argument('--threads', type=int, default=1)
    pr.add_argument('--output', type=Path)
    ev = sub.add_parser('evaluate', help='Repeat an immutable released-model evaluation')
    ev.add_argument('--model', choices=['stylometric', 'transformer'], required=True)
    ev.add_argument('--model-dir', type=Path, required=True)
    ev.add_argument('--model-name', required=True)
    ev.add_argument('--manifest', type=Path, default=ROOT / 'reports/split-manifest.json')
    ev.add_argument('--data-root', type=Path, default=ROOT)
    ev.add_argument('--split', choices=['selection', 'calibration', 'locked_holdout'], default='locked_holdout')
    ev.add_argument('--freeze', type=Path, default=ROOT / 'reports/evaluation-freeze.json')
    ev.add_argument('--output', type=Path, required=True)
    ev.add_argument('--batch-size', type=int, default=8)
    ev.add_argument('--threads', type=int, default=1)
    bm = sub.add_parser('benchmark', help='Measure complete pair prediction after model loading')
    bm.add_argument('--model', choices=['stylometric', 'transformer'], required=True)
    bm.add_argument('--model-dir', type=Path, required=True)
    bm.add_argument('--model-name', required=True)
    bm.add_argument('--freeze', type=Path, default=ROOT / 'reports/evaluation-freeze.json')
    bm.add_argument('--manifest', type=Path, default=ROOT / 'reports/split-manifest.json')
    bm.add_argument('--data-root', type=Path, default=ROOT)
    bm.add_argument('--samples', type=int, default=32)
    bm.add_argument('--warmup', type=int, default=5)
    bm.add_argument('--repetitions', type=int, default=3)
    bm.add_argument('--threads', type=int, default=1)
    bm.add_argument('--output', type=Path)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == 'doctor':
            from .runtime import doctor
            result = doctor()
            write_result(result, args.output)
            return 0 if result['ready'] else 1
        if args.command == 'download':
            from scripts.download_artifacts import install_artifacts, load_descriptor, verify_artifacts
            descriptor = load_descriptor(args.manifest)
            name = args.model
            if name == 'transformer':
                frozen = json.loads((ROOT / 'reports/evaluation-freeze.json').read_text(encoding='utf-8'))
                name = frozen['config']['selected_transformer']
            names = ['stylometric', 'ce', 'custom_asl'] if name == 'all' else [name]
            results = {}
            for selected in names:
                prefix = 'models/' + selected + '/'
                entries = [{**entry, 'path': entry['path'][len(prefix):]}
                           for entry in descriptor['artifacts'] if entry['path'].startswith(prefix)]
                if not entries:
                    raise ValueError('Requested model is absent from the verified public descriptor: ' + selected)
                part = {**descriptor, 'artifacts': entries}
                destination = args.output / 'models' / selected
                results[selected] = verify_artifacts(part, destination) if args.verify_only else install_artifacts(part, destination)
            write_result({'verified': True, 'models': results})
        elif args.command in {'verify-data', 'prepare-split'}:
            from .data import build_split_manifest, load_manifest, load_records, save_manifest, validate_manifest
            if args.command == 'prepare-split':
                manifest = build_split_manifest(args.data_root)
                save_manifest(manifest, args.output)
            else:
                manifest = load_manifest(args.manifest, args.data_root, verify_sources=True)
            result = validate_manifest(manifest, args.data_root, verify_sources=True)
            load_records(args.data_root, manifest, 'train')
            write_result(result if result is not None else {'verified': True, 'manifest': str(args.output if args.command == 'prepare-split' else args.manifest)})
        elif args.command == 'predict':
            from .runtime import MAX_TEXT_BYTES, Runtime
            if args.model_name:
                model_name = args.model_name
            elif args.model == 'stylometric':
                model_name = 'stylometric'
            else:
                frozen = json.loads(args.freeze.read_text(encoding='utf-8'))
                model_name = frozen['config']['selected_transformer']
            model_dir = args.model_dir or ROOT / 'artifacts/runtime/models' / model_name
            texts = []
            for side in ('1', '2'):
                filename = getattr(args, 'text_file_' + side)
                if filename is not None:
                    if not filename.is_file() or filename.stat().st_size > MAX_TEXT_BYTES:
                        raise ValueError('Provide an existing UTF-8 passage file of at most2MiB')
                    with filename.open('rb') as stream:
                        raw = stream.read(MAX_TEXT_BYTES + 1)
                    if len(raw) > MAX_TEXT_BYTES:
                        raise ValueError('Passage file exceeds2MiB')
                    text = raw.decode('utf-8')
                else:
                    text = getattr(args, 'text_' + side)
                texts.append(text if text is not None else input('English passage ' + side + ': '))
            result = Runtime(model_dir, args.model, args.threads, model_name=model_name,
                             freeze_path=args.freeze).predict_pair(*texts)
            write_result(result, args.output)
        elif args.command == 'evaluate':
            from .runtime import Runtime
            from .evaluation import evaluate_model
            model = Runtime(args.model_dir, args.model, args.threads, model_name=args.model_name,
                            freeze_path=args.freeze, manifest_path=args.manifest)
            result = evaluate_model(model.backend, args.data_root, args.manifest, args.split, args.output,
                                    model.calibration, args.model_dir, args.model_name,
                                    freeze_path=args.freeze, batch_size=args.batch_size)
            write_result(result)
        elif args.command == 'benchmark':
            import numpy as np
            from .data import load_records
            from .runtime import Runtime
            if min(args.samples, args.repetitions) < 1 or args.warmup < 0:
                raise ValueError('Positive sample/repetition counts and nonnegative warmup are required')
            records = load_records(args.data_root, args.manifest, 'selection')[:args.samples]
            if len(records) != args.samples:
                raise ValueError('Not enough declared selection records')
            begin = time.perf_counter()
            model = Runtime(args.model_dir, args.model, args.threads, model_name=args.model_name,
                            freeze_path=args.freeze, manifest_path=args.manifest)
            load_seconds = time.perf_counter() - begin
            for i in range(args.warmup):
                model.predict([records[i % len(records)]], batch_size=1)
            elapsed = []
            for _ in range(args.repetitions):
                for record in records:
                    start = time.perf_counter_ns()
                    model.predict([record], batch_size=1)
                    elapsed.append((time.perf_counter_ns() - start) / 1e6)
            files = [p for p in args.model_dir.rglob('*') if p.is_file()]
            result = {'device': 'cpu', 'model': args.model, 'threads': args.threads,
                      'platform': platform.platform(), 'processor': platform.processor(),
                      'samples': args.samples, 'warmup': args.warmup, 'repetitions': args.repetitions,
                      'batch_size': 1, 'pair_prediction_p50_ms': float(np.percentile(elapsed, 50)),
                      'pair_prediction_p95_ms': float(np.percentile(elapsed, 95)),
                      'model_load_seconds': load_seconds, 'directory_bytes': sum(p.stat().st_size for p in files),
                      'record_ids': [r['id'] for r in records],
                      'timing_scope': 'Complete frozen pair prediction: normalization, tokenization/features, selected orientation, model and calibration. Imports/model load/input-file reads excluded.'}
            write_result(result, args.output)
        return 0
    except (OSError, ValueError, RuntimeError, ImportError, KeyError) as exc:
        print(json.dumps({'ok': False, 'error': str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1
