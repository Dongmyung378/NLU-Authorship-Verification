import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.download_artifacts import download_file, load_descriptor, verify_artifacts


class Response(io.BytesIO):
    status = 200


class DownloadTests(unittest.TestCase):
    def entry(self, data, path='models/weights.bin'):
        return {'path': path, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}

    def test_multipart_reconstructs_exact_frozen_bytes(self):
        first, second = b'first content', b'second content'
        entry = self.entry(first + second)
        entry['parts'] = [{**self.entry(data), 'url': 'https://example.test/' + str(i)} for i, data in enumerate((first, second))]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            descriptor = root / 'descriptor.json'
            descriptor.write_text(json.dumps({'schema_version': 1, 'artifacts': [entry]}))
            load_descriptor(descriptor)
            installed = root / 'installed'
            target = installed / entry['path']
            with patch('scripts.download_artifacts.urllib.request.build_opener') as opener:
                opener.return_value.open.side_effect = [Response(first), Response(second)]
                download_file(entry, target)
            self.assertEqual(target.read_bytes(), first + second)
            self.assertEqual(verify_artifacts(descriptor, installed),
                             {'verified': True, 'files': 1, 'bytes': len(first + second)})

    def test_corrupt_part_cannot_pass_complete_file_hash(self):
        entry = self.entry(b'expected')
        entry['parts'] = [{**self.entry(b'expected'), 'url': 'https://example.test/part'}]
        with tempfile.TemporaryDirectory() as directory:
            with patch('scripts.download_artifacts.urllib.request.build_opener') as opener:
                opener.return_value.open.return_value = Response(b'corrupted')
                with self.assertRaises(ValueError):
                    download_file(entry, Path(directory) / 'weights')

    def test_descriptor_rejects_escape_and_bad_part_totals(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'descriptor.json'
            entry = {**self.entry(b'x', '../escape'), 'url': 'https://example.test/file'}
            path.write_text(json.dumps({'schema_version': 1, 'artifacts': [entry]}))
            with self.assertRaises(ValueError):
                load_descriptor(path)
            entry = self.entry(b'xx')
            entry['parts'] = [{**self.entry(b'x'), 'url': 'https://example.test/part'}]
            path.write_text(json.dumps({'schema_version': 1, 'artifacts': [entry]}))
            with self.assertRaises(ValueError):
                load_descriptor(path)
