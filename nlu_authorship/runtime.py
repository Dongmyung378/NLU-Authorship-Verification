"""CPU prediction using hash-checked artifacts and frozen calibration."""
from __future__ import annotations

import importlib.metadata
import json
import math
from pathlib import Path
import platform
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAX_TEXT_BYTES = 2 * 1024 * 1024


def configure_cpu(threads=1):
    if not isinstance(threads, int) or isinstance(threads, bool) or threads < 1:
        raise ValueError('CPU threads must be a positive integer')
    import torch
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        if torch.get_num_interop_threads() != 1:
            raise RuntimeError('Use a fresh process to configure CPU thread counts')
    return {'device': 'cpu', 'threads': threads, 'torch_version': torch.__version__}


def doctor():
    problems, versions = [], {}
    for line in (PROJECT_ROOT / 'requirements-runtime.txt').read_text(encoding='utf-8').splitlines():
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        package, expected = line.strip().split('==', 1)
        try:
            actual = importlib.metadata.version(package)
            versions[package] = actual
            if actual != expected:
                problems.append(f'{package}: expected {expected}, found {actual}')
        except importlib.metadata.PackageNotFoundError:
            problems.append(f'{package}: missing')
    if sys.version_info[:2] != (3, 12) or sys.maxsize <= 2**32:
        problems.append('A 64-bit Python3.12 runtime is required')
    if sys.prefix == sys.base_prefix:
        problems.append('Use the isolated project virtual environment')
    operation_ok = False
    if not problems:
        import torch
        configure_cpu()
        operation_ok = bool(torch.equal(torch.tensor([[1., 2.]]) @ torch.tensor([[3.], [4.]]), torch.tensor([[11.]])))
        if not operation_ok:
            problems.append('CPU tensor operation failed')
    return {'ready': not problems, 'problems': problems, 'python': platform.python_version(),
            'platform': platform.platform(), 'device': 'cpu', 'isolated': sys.prefix != sys.base_prefix,
            'versions': versions, 'actual_cpu_tensor_operation': operation_ok,
            'models_installed_checked': False}


def checked_text(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError('Provide two non-empty English passages')
    if len(value.encode('utf-8')) > MAX_TEXT_BYTES:
        raise ValueError(f'Each passage must be at most {MAX_TEXT_BYTES} UTF-8 bytes')
    return value


class Runtime:
    def __init__(self, model_dir, kind='transformer', threads=1, *, model_name=None,
                 freeze_path=None, manifest_path=None):
        original = Path(model_dir).absolute()
        if original.is_symlink() or any(parent.is_symlink() for parent in original.parents):
            raise ValueError('Model directories cannot pass through symbolic links')
        self.model_dir = original.resolve(strict=True)
        if not self.model_dir.is_dir():
            raise ValueError('Provide a local released model directory')
        self.kind = kind
        from .evaluation import validate_freeze
        freeze_path = Path(freeze_path) if freeze_path is not None else PROJECT_ROOT / 'reports/evaluation-freeze.json'
        manifest_path = Path(manifest_path) if manifest_path is not None else PROJECT_ROOT / 'reports/split-manifest.json'
        model_name = model_name or self.model_dir.name
        frozen = validate_freeze(freeze_path, manifest_path, self.model_dir, model_name)
        self.model_name = model_name
        self.calibration = frozen['calibrations'][model_name]
        if self.calibration.get('schema_version') != 1 or self.calibration.get('fitting_split') != 'calibration':
            raise ValueError('Calibration must be fitted on the declared calibration split')
        threshold = float(self.calibration['threshold'])
        cutoff = float(self.calibration['abstention']['confidence_threshold'])
        if not math.isfinite(threshold) or not 0 <= threshold <= 1 or not math.isfinite(cutoff) or not 0 <= cutoff <= 1:
            raise ValueError('Released threshold and abstention cutoff must be finite and in[0,1]')
        configure_cpu(threads)
        if kind == 'transformer':
            from .models import load_transformer
            self.backend = load_transformer(self.model_dir, device='cpu')
        elif kind == 'stylometric':
            from .stylometric import load_stylometric
            self.backend = load_stylometric(self.model_dir)
            # Override loaded estimators in memory. The same requested CPU
            # budget applies to Torch, XGBoost and LightGBM without resaving
            # or changing the hash-bound model artifacts.
            self.backend.xgb.set_params(n_jobs=threads)
            self.backend.lgb.set_params(n_jobs=threads)
        else:
            raise ValueError('Model kind must be transformer or stylometric')

    def logits(self, records, batch_size=8, orientation='ab'):
        if orientation not in {'ab', 'ba'}:
            raise ValueError('Orientation must be ab or ba')
        if self.kind == 'transformer':
            return self.backend.predict_logits(records, batch_size=batch_size, orientation=orientation)
        from .metrics import probability_logits
        rows = records if orientation == 'ab' else [{**r, 'text_1': r['text_2'], 'text_2': r['text_1']} for r in records]
        return probability_logits(self.backend.predict_proba(rows, batch_size=batch_size))

    def predict(self, records, batch_size=8):
        from .calibration import apply_calibration
        ab = self.logits(records, batch_size)
        ba = self.logits(records, batch_size, 'ba') if self.calibration['inference_method'] == 'symmetric' else None
        scores = apply_calibration(ab, self.calibration, ba)
        threshold = self.calibration['threshold']
        cutoff = self.calibration['abstention']['confidence_threshold']
        labels = (scores >= threshold).astype(int)
        confidence = np.where(labels == 1, scores, 1 - scores)
        return [{'id': record.get('id', str(i)), 'prediction': int(label),
                 'decision': ('same_author' if label else 'different_authors') if accepted >= cutoff else 'uncertain',
                 'same_author_score': float(score), 'confidence_for_predicted_class': float(accepted),
                 'abstained': bool(accepted < cutoff), 'threshold': float(threshold),
                 'temperature': float(self.calibration['temperature']),
                 'inference_method': self.calibration['inference_method']}
                for i, (record, score, label, accepted) in enumerate(zip(records, scores, labels, confidence))]

    def predict_pair(self, text_1, text_2):
        return self.predict([{'id': 'input', 'text_1': checked_text(text_1), 'text_2': checked_text(text_2)}])[0]
