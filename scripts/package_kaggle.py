"""Package the exact declared code and data for one private Kaggle GPU run."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


_WRAPPER_TEMPLATE = r'''"""Verify the frozen ZIP or Kaggle-expanded input before GPU setup."""
from pathlib import Path, PurePosixPath
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import zipfile

PROJECT_NAME = 'NLU-Authorship-Verification'
EXPECTED_ARCHIVE_SHA256 = __ARCHIVE_SHA__
EXPECTED_ARCHIVE_BYTES = __ARCHIVE_BYTES__
PROFILE_ONLY = __PROFILE_ONLY__
THROUGHPUT_PROFILE = __THROUGHPUT_PROFILE__
EXPECTED_FILES = __INVENTORY__
SOURCE_BUNDLE = {'name': PROJECT_NAME, 'archive_name': 'nlu-source.zip',
                 'archive_sha256': EXPECTED_ARCHIVE_SHA256, 'archive_bytes': EXPECTED_ARCHIVE_BYTES,
                 'files': EXPECTED_FILES, 'kind': 'source'}
MODEL_BUNDLE = __MODEL_BUNDLE__


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def safe_relative(name):
    if not isinstance(name, str) or not name or '\\' in name or ':' in name or '\x00' in name:
        raise ValueError('Unsafe frozen source path')
    if PurePosixPath(name).is_absolute() or any(part in {'', '.', '..'} for part in name.split('/')):
        raise ValueError('Unsafe frozen source path')
    return Path(*PurePosixPath(name).parts)


def no_symlink_path(path, boundary):
    path, boundary = Path(path).absolute(), Path(boundary).absolute()
    if not path.is_relative_to(boundary):
        raise ValueError('Source path leaves its declared boundary')
    current = path
    while True:
        if current.is_symlink():
            raise ValueError('Symbolic links are forbidden in source input')
        if current == boundary:
            break
        current = current.parent
    return path


def verify_expanded(root, boundary, specification=SOURCE_BUNDLE):
    no_symlink_path(root, boundary)
    if not root.is_dir():
        raise ValueError('Expanded source root is not a directory')
    for name, entry in specification['files'].items():
        path = no_symlink_path(root / safe_relative(name), boundary)
        if not path.is_file() or not stat.S_ISREG(path.stat().st_mode):
            raise ValueError('Missing regular frozen source file: ' + name)
        if path.stat().st_size != entry['bytes'] or sha256_file(path) != entry['sha256']:
            raise ValueError('Frozen source size/SHA256 mismatch: ' + name)


def zip_inventory(bundle, specification=SOURCE_BUNDLE):
    expected = {specification['name'] + '/' + name: entry for name, entry in specification['files'].items()}
    files = {}
    names = set()
    for member in bundle.infolist():
        raw = member.filename.rstrip('/') if member.is_dir() else member.filename
        safe_relative(raw)
        if stat.S_ISLNK(member.external_attr >> 16):
            raise ValueError('ZIP source symbolic links are forbidden')
        if member.is_dir():
            continue
        if member.filename.casefold() in names or member.filename not in expected:
            raise ValueError('ZIP contains duplicated or unlisted frozen source files')
        names.add(member.filename.casefold())
        entry = expected[member.filename]
        if member.file_size != entry['bytes']:
            raise ValueError('ZIP source size mismatch: ' + member.filename)
        digest = hashlib.sha256()
        count = 0
        with bundle.open(member) as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                count += len(block)
                if count > entry['bytes']:
                    raise ValueError('ZIP source exceeded its declared file size')
                digest.update(block)
        if count != entry['bytes'] or digest.hexdigest() != entry['sha256']:
            raise ValueError('ZIP source SHA256 mismatch: ' + member.filename)
        files[member.filename] = member
    if set(files) != set(expected):
        raise ValueError('ZIP is missing declared frozen source files')
    return files


def verify_archive(path, boundary, specification=SOURCE_BUNDLE):
    no_symlink_path(path, boundary)
    if not path.is_file() or path.stat().st_size != specification['archive_bytes'] or sha256_file(path) != specification['archive_sha256']:
        raise ValueError('Input ZIP differs from the exact frozen source archive')
    with zipfile.ZipFile(path) as bundle:
        zip_inventory(bundle, specification)


def prepare_bundle(input_dir, target, specification):
    input_dir, target = Path(input_dir).absolute(), Path(target).absolute()
    if not input_dir.is_dir() or input_dir.is_symlink():
        raise ValueError('Input root must be a regular directory')
    if target.exists() or target.is_symlink():
        raise FileExistsError('Preserve existing working source; choose a fresh working path')
    for parent in target.parents:
        if parent.is_symlink():
            raise ValueError('Working source cannot pass through a symbolic link')
    expected_files = specification['files']
    for name, entry in expected_files.items():
        safe_relative(name)
        if not isinstance(entry['bytes'], int) or entry['bytes'] <= 0 or len(entry['sha256']) != 64:
            raise ValueError('Invalid embedded frozen file inventory')
    matched = []
    failed = []
    for archive in sorted(input_dir.rglob(specification['archive_name'])):
        try:
            verify_archive(archive, input_dir, specification)
            matched.append(('zip', archive))
        except (OSError, ValueError, zipfile.BadZipFile) as error:
            failed.append(str(error))
    for root in sorted(input_dir.rglob(specification['name'])):
        try:
            verify_expanded(root, input_dir, specification)
            matched.append(('expanded', root))
        except (OSError, ValueError) as error:
            failed.append(str(error))
    if len(matched) != 1:
        raise RuntimeError('Expected one unique verified frozen ' + specification['kind'] + ' candidate; found ' + str(len(matched)) +
                           '. Input checks: ' + '; '.join(failed[:4]))
    kind, selected = matched[0]
    target.mkdir(parents=True, exist_ok=False)
    root = target / specification['name']
    root.mkdir()
    if kind == 'zip':
        with zipfile.ZipFile(selected) as bundle:
            members = zip_inventory(bundle, specification)
            for name in expected_files:
                destination = root / safe_relative(name)
                destination.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(members[specification['name'] + '/' + name]) as source, destination.open('xb') as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
    else:
        for name in expected_files:
            source = no_symlink_path(selected / safe_relative(name), input_dir)
            destination = root / safe_relative(name)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with source.open('rb') as stream, destination.open('xb') as output:
                shutil.copyfileobj(stream, output, length=1024 * 1024)
    # Recheck copied bytes immediately before executing the bootstrap. No
    # unlisted file, environment, credential, notebook, or historical model is copied.
    verify_expanded(root, target, specification)
    receipt = {'stage': 'frozen-input-verified', 'bundle_kind': specification['kind'], 'input_kind': kind,
               'files': len(expected_files), 'archive_sha256': specification['archive_sha256'], 'profile_only': PROFILE_ONLY,
               'copied_inventory_sha256': hashlib.sha256(json.dumps(expected_files, sort_keys=True, separators=(',', ':')).encode()).hexdigest()}
    (target / 'input-verification.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf8')
    print(json.dumps(receipt), flush=True)
    return root, receipt


def prepare_source(input_dir, target):
    return prepare_bundle(input_dir, target, SOURCE_BUNDLE)


def prepare_stylometric_model(input_dir, target):
    if MODEL_BUNDLE is None:
        raise ValueError('This package does not declare a stylometric model')
    return prepare_bundle(input_dir, target, MODEL_BUNDLE)


def main():
    root, receipt = prepare_source(Path('/kaggle/input'), Path('/kaggle/working/nlu-source'))
    model_args = []
    if MODEL_BUNDLE is not None:
        model_dir, model_receipt = prepare_stylometric_model(Path('/kaggle/input'), Path('/kaggle/working/nlu-stylometric-model'))
        model_args = ['--stylometric-model-dir', str(model_dir)]
    result = subprocess.run([sys.executable, str(root / 'scripts/setup_kaggle_runtime.py'),
        '--project-root', str(root), '--runtime', '/tmp/nlu-authorship-runtime',
        '--metadata-file', '/kaggle/working/nlu-results/kaggle-runtime.json', '--',
        '-m', 'scripts.run_kaggle_experiment', '--repo-root', str(root),
        '--manifest', str(root / 'reports/split-manifest.json'), '--plan', str(root / 'configs/experiment-plan.json'),
        '--output', '/kaggle/working/nlu-results'] + model_args + (['--throughput-profile'] if THROUGHPUT_PROFILE else ['--profile-only'] if PROFILE_ONLY else ['--complete-release']), check=True)
    print(json.dumps({'stage': 'throughput-profile-complete' if THROUGHPUT_PROFILE else 'profile-complete' if PROFILE_ONLY else 'project-complete', 'exit_code': result.returncode}), flush=True)


if __name__ == '__main__':
    main()
'''

def _validate_bundle_record(record):
    """Validate a portable, case-safe byte inventory before embedding it."""
    import re
    from pathlib import PurePosixPath
    files = record.get('files')
    if not isinstance(files, dict) or not files:
        raise ValueError('A nonempty frozen per-file inventory is required')
    if not isinstance(record.get('archive_bytes'), int) or record['archive_bytes'] <= 0 or not re.fullmatch(r'[0-9a-f]{64}', record.get('archive_sha256', '')):
        raise ValueError('The frozen archive size/SHA256 are required')
    names = set()
    for name, entry in files.items():
        if not isinstance(name, str) or not name or '\\' in name or ':' in name or '\x00' in name or PurePosixPath(name).is_absolute() or any(part in {'', '.', '..'} for part in name.split('/')) or name.casefold() in names:
            raise ValueError('Unsafe or duplicated frozen source path')
        names.add(name.casefold())
        if not isinstance(entry.get('bytes'), int) or entry['bytes'] <= 0 or not re.fullmatch(r'[0-9a-f]{64}', entry.get('sha256', '')):
            raise ValueError('Every frozen source file needs a size and SHA256')
    return files


def render_wrapper(record, *, profile_only=False, throughput_profile=False):
    """Embed exact source and optional model inventories for both mount types."""
    if not isinstance(profile_only, bool) or not isinstance(throughput_profile, bool):
        raise ValueError('Profiling mode flags must be boolean')
    if profile_only and throughput_profile:
        raise ValueError('Capability and throughput profiles are mutually exclusive')
    profiling = profile_only or throughput_profile
    files = _validate_bundle_record(record)
    model_record = record.get('stylometric_model')
    if profiling and model_record is not None:
        raise ValueError('A profile-only package cannot include a stylometric model')
    model_bundle = None
    if model_record is not None:
        model_files = _validate_bundle_record(model_record)
        model_bundle = {'name': 'NLU-Stylometric-Model', 'archive_name': 'stylometric-model.zip',
                        'archive_sha256': model_record['archive_sha256'], 'archive_bytes': model_record['archive_bytes'],
                        'files': model_files, 'kind': 'stylometric-model'}
    return (_WRAPPER_TEMPLATE.replace('__ARCHIVE_SHA__', repr(record['archive_sha256']))
            .replace('__ARCHIVE_BYTES__', str(record['archive_bytes']))
            .replace('__INVENTORY__', repr(files))
            .replace('__MODEL_BUNDLE__', repr(model_bundle))
            .replace('__PROFILE_ONLY__', repr(profiling))
            .replace('__THROUGHPUT_PROFILE__', repr(throughput_profile)))


_STYLOMETRIC_REQUIRED = ('ensemble_config.json', 'xgb_model.json', 'lgb_model.pkl',
                         'tfidf_word.pkl', 'tfidf_char.pkl', 'tfidf_fw.pkl')


def _verified_stylometric_files(model_dir):
    """Verify the saved CPU ensemble without loading pickle or model weights."""
    from nlu_authorship.models import verify_artifact_inventory
    model_dir = Path(model_dir).absolute()
    if any(path.is_symlink() for path in (model_dir, *model_dir.parents)):
        raise ValueError('Stylometric model input cannot pass through symbolic links')
    model_dir = model_dir.resolve(strict=True)
    document = verify_artifact_inventory(model_dir, _STYLOMETRIC_REQUIRED)
    # This transport deliberately accepts the canonical CPU ensemble only.
    # A valid inventory of a historical transformer is not a stylometric model.
    allowed = set(_STYLOMETRIC_REQUIRED)
    if {entry['path'] for entry in document['files']} != allowed:
        raise ValueError('Only the canonical stylometric ensemble and its transforms may be packaged')
    files = {entry['path']: {'bytes': entry['bytes'], 'sha256': entry['sha256']}
             for entry in document['files']}
    inventory = model_dir / 'artifact-inventory.json'
    files['artifact-inventory.json'] = {'bytes': inventory.stat().st_size, 'sha256': sha256(inventory)}
    return model_dir, dict(sorted(files.items()))


def package(output, dataset_id, kernel_id, profile_only=False, stylometric_model=None, throughput_profile=False):
    from nlu_authorship.data import load_records
    from nlu_authorship.evaluation import source_inventory
    if not isinstance(profile_only, bool) or not isinstance(throughput_profile, bool):
        raise ValueError('Profiling mode flags must be boolean')
    if profile_only and throughput_profile:
        raise ValueError('Capability and throughput profiles are mutually exclusive')
    profiling = profile_only or throughput_profile
    if profiling and stylometric_model is not None:
        raise ValueError('A profile-only package cannot include a stylometric model')
    output = Path(output)
    if output.exists():
        raise FileExistsError('Preserve previous run package; choose a new output directory')
    model_dir, model_files = _verified_stylometric_files(stylometric_model) if stylometric_model is not None else (None, None)
    dataset = output / 'dataset'
    kernel = output / 'kernel'
    dataset.mkdir(parents=True)
    kernel.mkdir()
    manifest = ROOT / 'reports/split-manifest.json'
    load_records(ROOT, manifest, 'train')
    core = source_inventory(ROOT)
    files = sorted(set(core) | {'README.md', 'reports/split-manifest.json', 'data/train.csv', 'data/dev.csv'})
    archive = dataset / 'nlu-source.zip'
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
        for relative in files:
            bundle.write(ROOT / relative, 'NLU-Authorship-Verification/' + relative)
    record = {'schema_version': 1, 'archive_sha256': sha256(archive), 'archive_bytes': archive.stat().st_size,
              'files': {name: {'sha256': sha256(ROOT / name), 'bytes': (ROOT / name).stat().st_size} for name in files},
              'source_inventory': core, 'dataset_id': dataset_id, 'kernel_id': kernel_id,
              'private_dataset': True, 'gpu_enabled': True, 'profile_only': profiling, 'throughput_profile': throughput_profile,
              'profile_mode': 'throughput' if throughput_profile else 'capability' if profile_only else None,
              'excluded': ['credentials', 'git metadata', 'environments', 'weights', 'notebooks', 'unlabeled test/live', 'personal logs']}
    if model_dir is not None:
        model_archive = dataset / 'stylometric-model.zip'
        with zipfile.ZipFile(model_archive, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
            for name in model_files:
                bundle.write(model_dir / name, 'NLU-Stylometric-Model/' + name)
        # Reject an artifact changed between initial verification and packaging.
        if _verified_stylometric_files(model_dir)[1] != model_files:
            raise ValueError('Stylometric artifacts changed while packaging')
        record['stylometric_model'] = {'archive_name': model_archive.name, 'top_folder': 'NLU-Stylometric-Model',
                                     'archive_sha256': sha256(model_archive), 'archive_bytes': model_archive.stat().st_size,
                                     'files': model_files, 'artifact_inventory_verified': True,
                                     'training_device': 'cpu', 'transport_only': True}
        record['excluded'][3] = 'historical transformer weights'
    (output / 'source-package.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8', newline='\n')
    metadata = {'title': 'NLU document disjoint source 20261012', 'id': dataset_id,
                'licenses': [{'name': 'other'}],
                'description': 'Private input package for the owner-authorized NLU engineering experiment. Original data license is not reassigned.'}
    (dataset / 'dataset-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8', newline='\n')
    run = render_wrapper(record, profile_only=profile_only, throughput_profile=throughput_profile)
    (kernel / 'run-experiment.py').write_text(run, encoding='utf-8', newline='\n')
    # Kaggle can derive the kernel slug from title on creation. Keep both
    # representations aligned instead of allowing a hardcoded title to rename it.
    kernel_title = kernel_id.rsplit('/', 1)[-1].replace('-', ' ')
    kernel_metadata = {'id': kernel_id, 'title': kernel_title,
                       'code_file': 'run-experiment.py', 'language': 'python', 'kernel_type': 'script',
                       'is_private': True, 'enable_gpu': True, 'enable_internet': True,
                       'machine_shape': 'NvidiaTeslaT4', 'dataset_sources': [dataset_id],
                       'competition_sources': [], 'kernel_sources': [], 'model_sources': []}
    (kernel / 'kernel-metadata.json').write_text(json.dumps(kernel_metadata, indent=2) + '\n', encoding='utf-8', newline='\n')
    return record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--dataset-id', required=True)
    p.add_argument('--kernel-id', required=True)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--profile-only', action='store_true', help='Package only the GPU capability profile; do not train or evaluate final results')
    mode.add_argument('--throughput-profile', action='store_true', help='Package only a bounded training-throughput probe; discard all probe weights')
    p.add_argument('--stylometric-model', type=Path, help='Verified saved CPU ensemble to reuse in the full GPU job')
    args = p.parse_args()
    record = package(args.output, args.dataset_id, args.kernel_id, profile_only=args.profile_only, stylometric_model=args.stylometric_model,
                     throughput_profile=args.throughput_profile)
    print(json.dumps({'ready': True, 'archive_bytes': record['archive_bytes'], 'archive_sha256': record['archive_sha256'], 'files': len(record['files'])}))


if __name__ == '__main__':
    main()
