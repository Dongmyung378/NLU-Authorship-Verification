import copy
from pathlib import Path
import tempfile
import unittest

from nlu_authorship.data import (build_split_manifest, canonical_sha256, load_records,
                                 save_manifest, validate_manifest, audit_tokenized_input_overlap)
from nlu_authorship.normalization import identity_views, normalize_stylometric_text
from nlu_fixtures import write_dataset


class DataProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = write_dataset(Path(self.temp.name))
        self.manifest = build_split_manifest(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def test_numeric_labels_reversed_pairs_and_conflicts(self):
        reasons = [r["reason"] for r in self.manifest["excluded_rows"]]
        self.assertEqual(reasons.count("consistent_unordered_pair_duplicate"), 1)
        self.assertEqual(reasons.count("conflicting_normalized_pair_labels"), 2)
        self.assertEqual(reasons.count("empty_normalized_document"), 1)
        self.assertEqual(self.manifest["retained_rows"], 102)
        self.assertTrue(all(r["label"] in (0, 1) for r in self.manifest["records"]))

    def test_unicode_runtime_identity_groups_before_assignment(self):
        self.assertEqual(normalize_stylometric_text("e\u0301e\u0301e\u0301e\u0301"), normalize_stylometric_text("éé"))
        self.assertNotEqual(identity_views("e\u0301e\u0301e\u0301e\u0301")["legacy_stylometric"], identity_views("éé")["legacy_stylometric"])
        rows = [r for split in ("train", "selection", "calibration", "locked_holdout") for r in load_records(self.root, self.manifest, split)]
        unicode_rows = [r for r in rows if r["text_1"] in {"e\u0301e\u0301e\u0301e\u0301", "éé"}]
        self.assertEqual(len(unicode_rows), 2)
        self.assertEqual(len({r["group_id"] for r in unicode_rows}), 1)

    def test_source_change_and_manifest_change_are_rejected(self):
        damaged = copy.deepcopy(self.manifest)
        damaged["records"][0]["label"] = 1 - damaged["records"][0]["label"]
        with self.assertRaisesRegex(ValueError, "checksum"):
            validate_manifest(damaged)
        path = self.root / "data" / "train.csv"
        path.write_bytes(path.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "checksum"):
            load_records(self.root, self.manifest, "train")

    def test_deterministic_immutable_manifest_and_record_order(self):
        self.assertEqual(self.manifest, build_split_manifest(self.root))
        path = self.root / "reports" / "split-manifest.json"
        save_manifest(self.manifest, path)
        self.assertNotIn(b"\r\n", path.read_bytes())
        records = load_records(self.root, path, "selection")
        self.assertEqual([r["id"] for r in records], [r["id"] for r in self.manifest["records"] if r["split"] == "selection"])
        other = build_split_manifest(self.root, seed=379)
        with self.assertRaises(FileExistsError):
            save_manifest(other, path)

    def test_forged_cross_component_assignments_rejected(self):
        damaged = copy.deepcopy(self.manifest)
        row = next(r for r in damaged["records"] if r["group_id"] == damaged["records"][-1]["group_id"])
        row["split"] = "train" if row["split"] != "train" else "selection"
        damaged["manifest_sha256"] = canonical_sha256({k: v for k, v in damaged.items() if k != "manifest_sha256"})
        with self.assertRaises(ValueError):
            validate_manifest(damaged)

    def test_truncated_token_fingerprint_detects_cross_split_input_identity(self):
        class Tokenizer:
            def __call__(self, first, second, **kwargs):
                self.kwargs = kwargs
                # A tiny tokenizer fixture intentionally discards suffixes.
                values = [[len(a.split()[0]), len(b.split()[0])] for a, b in zip(first, second)]
                return {"input_ids": values, "attention_mask": [[1, 1] for _ in values]}
        tokenizer = Tokenizer()
        report = audit_tokenized_input_overlap(tokenizer, {
            "train": [{"id": "a", "text_1": "Common different ending", "text_2": "Shared train ending"}],
            "locked_holdout": [{"id": "b", "text_1": "Common novel ending", "text_2": "Shared other ending"}]})
        self.assertGreater(report["train_locked_holdout_shared_fingerprints"], 0)
        self.assertEqual(report["affected_record_ids"]["locked_holdout"], ["b"])
        self.assertFalse(report["labels_used"])
        self.assertEqual(tokenizer.kwargs["truncation"], "longest_first")
