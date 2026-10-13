"""Offline, input-only audit of the frozen RoBERTa-Large pair-length policy.

Run with the project's isolated CPU Python. Supply a local snapshot containing
the five pinned tokenizer/config assets; this helper never downloads weights or
assets, runs a model, changes a split, or chooses a configuration.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time


MODEL_ID = 'FacebookAI/roberta-large'
REVISION = '722cf37b1afa9454edce342e7895e588b6ff1d59'
ASSET_HASHES = {
    'config.json': '82ba49810e6441735e696033ac2512ee09da555507b2917ac3865b202d592cc3',
    'merges.txt': '1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5',
    'tokenizer.json': '847bbeab6174d66a88898f729d52fa8d355fafe1bea101cf960dd404581df70e',
    'tokenizer_config.json': '994f46754c5bf4014f1aa92d34b1374319c3a6b3f702105cd5b742beaecd18ce',
    'vocab.json': '9e7f63c2d15d666b52e21d250d2e513b87c9b713cfa6987a82ed89e5e6e50655',
}
SPLITS = ('train', 'selection', 'calibration', 'locked_holdout')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                                     allow_nan=False).encode('utf-8')).hexdigest()


def source_inventory(root):
    paths = set((root / 'nlu_authorship').rglob('*.py'))
    paths.update((root / 'scripts').rglob('*.py'))
    paths.update((root / 'scripts').rglob('*.ps1'))
    paths.update((root / 'configs').rglob('*.json'))
    paths.update(root.glob('requirements*.txt'))
    paths.update(root.glob('launch-nlu.*'))
    paths.update(p for p in (root / 'run.ps1', root / 'pyproject.toml') if p.is_file())
    return {p.relative_to(root).as_posix(): sha(p) for p in sorted(paths) if p.is_file()}


def assets(directory):
    result = {}
    for name, expected in ASSET_HASHES.items():
        path = directory / name
        if not path.is_file() or sha(path) != expected:
            raise ValueError('Local pinned tokenizer asset differs: ' + name)
        result[name] = {'bytes': path.stat().st_size, 'sha256': expected}
    return result


def describe(lengths, minimum_words, max_length):
    import numpy as np
    values = np.asarray(lengths, dtype=np.int64)
    if not len(values) or len(minimum_words) != len(values):
        raise ValueError('A split must contain complete paired inputs')
    above = int((values > max_length).sum())
    return {'count': len(values), 'above_max_length_count': above, 'truncation_fraction': above / len(values),
            'pretruncation_pair_tokens_including_specials': {
                'minimum': int(values.min()), 'mean': float(values.mean()),
                'percentiles': {str(p): float(np.percentile(values, p, method='linear')) for p in (5, 25, 50, 75, 90, 95, 99)},
                'maximum': int(values.max()), 'percentile_method': 'linear'},
            'minimum_cleaned_word_bin_counts': {
                'under_50': sum(value < 50 for value in minimum_words),
                '50_to_199': sum(50 <= value < 200 for value in minimum_words),
                '200_plus': sum(value >= 200 for value in minimum_words)},
            'minimum_cleaned_word_bin_policy': 'Minimum whitespace word count of the two NFC/RoBERTa-cleaned passages, before token truncation.'}


def audit(repo_root, manifest_path, tokenizer_dir, output):
    repo_root, manifest_path = Path(repo_root).resolve(strict=True), Path(manifest_path).resolve(strict=True)
    tokenizer_dir, output = Path(tokenizer_dir).resolve(strict=True), Path(output)
    if output.exists():
        raise FileExistsError('Preserve previous measured reports; choose a new output path')
    before = source_inventory(repo_root)
    manifest_hash = sha(manifest_path)
    asset_inventory = assets(tokenizer_dir)
    helper_hash = sha(Path(__file__))
    plan = json.loads((repo_root / 'configs/experiment-plan.json').read_text(encoding='utf-8'))
    if plan['pretrained']['model_id'] != MODEL_ID or plan['pretrained']['revision'] != REVISION or plan['transformer']['max_length'] != 512:
        raise ValueError('The audit must describe the frozen RoBERTa-Large512-token protocol')
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_IMPLICIT_TOKEN='1',
                      TOKENIZERS_PARALLELISM='false')
    sys.path.insert(0, str(repo_root))
    from nlu_authorship.data import load_manifest, text_sha256
    from nlu_authorship.normalization import normalize_roberta_text
    from transformers import AutoTokenizer
    from transformers.utils import logging as transformer_logging
    transformer_logging.set_verbosity_error()
    started = time.perf_counter()
    manifest = load_manifest(manifest_path)
    originals = {}
    for name, source in manifest['source_csvs'].items():
        path = repo_root / 'data' / source['filename']
        if path.stat().st_size != source['bytes'] or sha(path) != source['sha256']:
            raise ValueError('Source CSV changed')
        with path.open(encoding='utf-8-sig', newline='') as stream:
            reader = csv.DictReader(stream)
            for index, row in enumerate(reader):
                # Labels are deliberately not accessed by this input audit.
                originals[f'{name}:{index:06d}'] = (row['text_1'], row['text_2'])
    records = []
    unique = {}
    ordered_inputs = []
    for record in manifest['records']:
        raw_a, raw_b = originals[record['id']]
        if text_sha256(raw_a) != record['text_1_sha256'] or text_sha256(raw_b) != record['text_2_sha256']:
            raise ValueError('A retained input differs from its manifest')
        a, b = normalize_roberta_text(raw_a), normalize_roberta_text(raw_b)
        unique.setdefault(a, None); unique.setdefault(b, None)
        records.append((record['split'], a, b))
        ordered_inputs.append({'id': record['id'], 'split': record['split'], 'group_id': record['group_id'],
                               'cleaned_text_1_sha256': text_sha256(a), 'cleaned_text_2_sha256': text_sha256(b)})
    del originals
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True, trust_remote_code=False)
    pair_specials = int(tokenizer.num_special_tokens_to_add(pair=True))
    if pair_specials < 1:
        raise ValueError('Tokenizer has no declared pair special tokens')
    texts = sorted(unique)
    for offset in range(0, len(texts), 64):
        batch = texts[offset:offset + 64]
        encoded = tokenizer(batch, add_special_tokens=False, truncation=False, padding=False,
                            return_attention_mask=False, return_token_type_ids=False, verbose=False)
        if len(encoded['input_ids']) != len(batch):
            raise ValueError('Tokenizer did not return every cleaned passage')
        for text, tokens in zip(batch, encoded['input_ids']):
            unique[text] = (len(tokens), len(text.split()))
    lengths = {name: [] for name in SPLITS}
    words = {name: [] for name in SPLITS}
    formula_checks = 0
    for offset in range(0, len(records), 64):
        batch = records[offset:offset + 64]
        encoded = tokenizer([r[1] for r in batch], [r[2] for r in batch], add_special_tokens=True,
                            truncation=False, padding=False, return_attention_mask=False,
                            return_token_type_ids=False, verbose=False)
        if len(encoded['input_ids']) != len(batch):
            raise ValueError('Tokenizer did not return every actual pair')
        for (split, a, b), tokens in zip(batch, encoded['input_ids']):
            formula = unique[a][0] + unique[b][0] + pair_specials
            if len(tokens) != formula:
                raise ValueError('Actual pair encoding differs from the special-token length formula')
            lengths[split].append(len(tokens))
            words[split].append(min(unique[a][1], unique[b][1]))
            formula_checks += 1
        if (offset // 64 + 1) % 64 == 0:
            print(json.dumps({'stage': 'input-only-pair-length-audit', 'pairs_checked': formula_checks}), flush=True)
    summaries = {name: describe(lengths[name], words[name], 512) for name in SPLITS}
    if {name: row['count'] for name, row in summaries.items()} != manifest['counts']:
        raise ValueError('Audited split counts differ from the immutable manifest')
    pooled = describe([value for name in SPLITS for value in lengths[name]],
                      [value for name in SPLITS for value in words[name]], 512)
    if source_inventory(repo_root) != before or sha(manifest_path) != manifest_hash or assets(tokenizer_dir) != asset_inventory or sha(Path(__file__)) != helper_hash:
        raise ValueError('Source, manifest, tokenizer or audit helper changed during measurement')
    report = {'schema_version': 1, 'status': 'completed', 'input_only': True,
              'tokenizer_repository': MODEL_ID, 'tokenizer_revision': REVISION, 'tokenizer_class': type(tokenizer).__name__,
              'tokenizer_assets': asset_inventory, 'model_max_length': 512, 'pair_special_token_count': pair_specials,
              'pair_truncation_policy_in_frozen_runtime': 'longest_first', 'audit_tokenization_truncation': False,
              'calculation': 'Actual full pair tokenization; independently verified as token_count(cleanedA,no_specials)+token_count(cleanedB,no_specials)+num_special_tokens_to_add(pair=True).',
              'actual_pair_formula_checks': formula_checks, 'unique_executed_cleaned_passages': len(unique),
              'normalization': 'Existing normalize_roberta_text: NFC guard followed by the declared RoBERTa cleanup.',
              'ordered_cleaned_input_identity_sha256': canonical(ordered_inputs),
              'manifest_sha256': manifest['manifest_sha256'], 'manifest_file_sha256': manifest_hash,
              'source_csvs': manifest['source_csvs'], 'source_inventory': before,
              'source_inventory_sha256': canonical(before), 'audit_helper_sha256': helper_hash,
              'public_reference_helper': 'docs/reference-code/token_length_audit.py',
              'packages': {name: importlib.metadata.version(name) for name in ('transformers', 'tokenizers', 'numpy')},
              'elapsed_seconds': time.perf_counter() - started, 'all_retained_pairs': pooled, 'by_split': summaries,
              'labels_used': False, 'label_metrics_computed': False, 'model_weights_loaded': False,
              'model_inference_performed': False, 'remote_downloads_performed': False,
              'source_manifest_tokenizer_unchanged': True, 'split_or_configuration_reselected': False,
              'scope': 'Input length and support diagnostics for the already frozen 512-token policy. A length above 512 indicates token truncation under that policy; it does not measure lost meaning, label quality, prediction accuracy or new-author generalization. These diagnostics do not change the active experiment.'}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--tokenizer', type=Path, required=True, help='Offline pinned snapshot directory with the five tokenizer/config files')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps({'stage': 'input-only-token-length-audit-started', 'pid': os.getpid()}), flush=True)
    report = audit(args.repo_root, args.manifest, args.tokenizer, args.output)
    print(json.dumps({'status': report['status'], 'pairs': report['all_retained_pairs']['count'],
                      'count_above512': report['all_retained_pairs']['above_max_length_count'],
                      'truncation_fraction': report['all_retained_pairs']['truncation_fraction'],
                      'elapsed_seconds': report['elapsed_seconds']}), flush=True)


if __name__ == '__main__':
    main()
