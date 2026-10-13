"""Reproducible document-disjoint labeled-pair splits without author-ID claims."""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

from .normalization import identity_views, normalize_roberta_text

SPLIT_NAMES = ("train", "selection", "calibration", "locked_holdout")
DEFAULT_RATIOS = (.70, .10, .05, .15)


def canonical_sha256(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class _UnionFind:
    def __init__(self, count):
        self.parents = list(range(count))
        self.sizes = [1] * count

    def find(self, i):
        while self.parents[i] != i:
            self.parents[i] = self.parents[self.parents[i]]
            i = self.parents[i]
        return i

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            if self.sizes[a] < self.sizes[b]:
                a, b = b, a
            self.parents[b] = a
            self.sizes[a] += self.sizes[b]


def _read_csv(path: Path, name: str) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if set(reader.fieldnames or []) != {"text_1", "text_2", "label"}:
            raise ValueError("Labeled CSV requires exactly text_1, text_2 and label")
        for index, row in enumerate(reader):
            label = float(row["label"])
            if label not in (0.0, 1.0) or row["text_1"] is None or row["text_2"] is None:
                raise ValueError("Invalid binary label or missing passage")
            rows.append({"id": f"{name}:{index:06d}", "source": name, "row_index": index,
                         "text_1": row["text_1"], "text_2": row["text_2"], "label": int(label)})
    return rows


def _sources(root: Path):
    rows, sources = [], {}
    for name in ("train", "dev"):
        path = root / "data" / f"{name}.csv"
        selected = _read_csv(path, name)
        sources[name] = {"filename": path.name, "bytes": path.stat().st_size,
                         "sha256": file_sha256(path), "rows": len(selected)}
        rows.extend(selected)
    return rows, sources


def build_split_manifest(repo_root, seed: int = 378, ratios=DEFAULT_RATIOS) -> dict:
    if len(ratios) != 4 or any(r <= 0 for r in ratios) or abs(sum(ratios) - 1) > 1e-9:
        raise ValueError("Four positive ratios totaling one are required")
    rows, sources = _sources(Path(repo_root))
    original_count = len(rows)
    texts = sorted({r[k] for r in rows for k in ("text_1", "text_2")})
    views = {text: identity_views(text) for text in texts}
    exclusions, eligible = [], []
    for row in rows:
        empty = sorted(mode for mode in views[row["text_1"]]
                       if not views[row["text_1"]][mode] or not views[row["text_2"]][mode])
        if empty:
            exclusions.append({"id": row["id"], "label": row["label"], "reason": "empty_normalized_document", "normalizations": empty})
        else:
            eligible.append(row)
    rows = eligible
    texts = sorted({r[k] for r in rows for k in ("text_1", "text_2")})
    ids = {text: i for i, text in enumerate(texts)}
    alias = _UnionFind(len(texts))
    for mode in ("exact", "nfc_whitespace", "legacy_stylometric", "legacy_roberta", "runtime_stylometric", "runtime_roberta"):
        first = {}
        for text in texts:
            key = text_sha256(views[text][mode])
            if not views[text][mode]:
                continue
            if key in first:
                alias.union(ids[text], first[key])
            else:
                first[key] = ids[text]
    members = defaultdict(list)
    for text in texts:
        members[alias.find(ids[text])].append(text_sha256(text))
    alias_keys = {group: canonical_sha256(sorted(values)) for group, values in members.items()}
    pairs = [(alias_keys[alias.find(ids[r["text_1"]])], alias_keys[alias.find(ids[r["text_2"]])]) for r in rows]
    pair_rows = defaultdict(list)
    for index, pair in enumerate(pairs):
        pair_rows[tuple(sorted(pair))].append(index)
    keep, conflict_groups = [], []
    for pair, indexes in sorted(pair_rows.items()):
        pair_id = canonical_sha256(pair)
        labels = {rows[i]["label"] for i in indexes}
        if len(labels) > 1:
            conflict_groups.append(pair_id)
            exclusions.extend({"id": rows[i]["id"], "label": rows[i]["label"], "reason": "conflicting_normalized_pair_labels", "pair_group": pair_id} for i in indexes)
        else:
            chosen = min(indexes, key=lambda i: rows[i]["id"])
            keep.append(chosen)
            exclusions.extend({"id": rows[i]["id"], "label": rows[i]["label"], "reason": "consistent_unordered_pair_duplicate", "kept_record_id": rows[chosen]["id"], "pair_group": pair_id} for i in indexes if i != chosen)
    keep.sort()
    rows, pairs = [rows[i] for i in keep], [pairs[i] for i in keep]
    keys = sorted({key for pair in pairs for key in pair})
    key_ids = {key: i for i, key in enumerate(keys)}
    components = _UnionFind(len(keys))
    for a, b in pairs:
        components.union(key_ids[a], key_ids[b])
    component_docs = defaultdict(list)
    for key in keys:
        component_docs[components.find(key_ids[key])].append(key)
    component_keys = {group: canonical_sha256(sorted(values)) for group, values in component_docs.items()}
    groups = defaultdict(list)
    for index, (a, _) in enumerate(pairs):
        groups[component_keys[components.find(key_ids[a])]].append(index)
    total = Counter(r["label"] for r in rows)
    targets = {name: {label: total[label] * ratio for label in (0, 1)} for name, ratio in zip(SPLIT_NAMES, ratios)}
    counts = {name: Counter() for name in SPLIT_NAMES}
    assigned = {}
    for group in sorted(groups, key=lambda g: (-len(groups[g]), canonical_sha256([seed, g]))):
        labels = Counter(rows[i]["label"] for i in groups[group])
        def cost(name):
            return sum(((counts[name][label] + labels[label] - targets[name][label]) ** 2
                        - (counts[name][label] - targets[name][label]) ** 2) / max(1.0, targets[name][label]) for label in (0, 1))
        split = min(SPLIT_NAMES, key=lambda name: (cost(name), SPLIT_NAMES.index(name)))
        assigned[group] = split
        counts[split].update(labels)
    records = []
    for row, pair in zip(rows, pairs):
        group = component_keys[components.find(key_ids[pair[0]])]
        records.append({"id": row["id"], "source": row["source"], "row_index": row["row_index"], "label": row["label"],
                        "text_1_sha256": text_sha256(row["text_1"]), "text_2_sha256": text_sha256(row["text_2"]),
                        "document_group_1": pair[0], "document_group_2": pair[1], "group_id": group,
                        "unordered_pair_group": canonical_sha256(sorted(pair)), "split": assigned[group]})
    manifest = {"schema_version": 1, "protocol": "nlu-document-components-v1", "seed": seed,
                "ratios": dict(zip(SPLIT_NAMES, ratios)), "source_csvs": sources,
                "identity_policy": "Transitive exact, NFC-whitespace, legacy raw cleanup and actual NFC-then-cleanup RoBERTa/stylometric equivalence; all labeled pairs connect documents regardless of label.",
                "quality_policy": "Reject empty cleanup results; quarantine all conflicting unordered-pair labels without changing labels; retain smallest original record ID per consistent unordered pair.",
                "assignment": "Largest components first; SHA seed tie breaking; greedy squared class-target deviation; no model scores.",
                "original_labeled_rows": original_count, "retained_rows": len(records), "excluded_rows": exclusions,
                "conflicting_pair_groups": sorted(conflict_groups), "records": records,
                "counts": {name: sum(counts[name].values()) for name in SPLIT_NAMES},
                "class_counts": {name: {str(label): counts[name][label] for label in (0, 1)} for name in SPLIT_NAMES},
                "component_counts": {name: sum(split == name for split in assigned.values()) for name in SPLIT_NAMES},
                "largest_component_rows": max(map(len, groups.values())),
                "historical_exposure": "Both original train and dev were used historically; locked_holdout denotes exclusion from this new model training and selection only.",
                "author_or_domain_separation_verified": False, "unlabeled_test_used": False}
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    validate_manifest(manifest, repo_root, verify_sources=True)
    return manifest


def read_manifest(value) -> dict:
    return json.loads(Path(value).read_text(encoding="utf-8")) if isinstance(value, (str, Path)) else dict(value)


def validate_manifest(manifest, repo_root=None, *, verify_sources=False):
    if manifest.get("schema_version") != 1 or manifest.get("protocol") != "nlu-document-components-v1":
        raise ValueError("Unsupported split manifest")
    if manifest.get("manifest_sha256") != canonical_sha256({k: v for k, v in manifest.items() if k != "manifest_sha256"}):
        raise ValueError("Split manifest checksum mismatch")
    records = manifest.get("records", [])
    if not records or len({r["id"] for r in records}) != len(records):
        raise ValueError("Empty manifest or duplicate record IDs")
    exclusions = manifest.get("excluded_rows", [])
    if len({r["id"] for r in exclusions}) != len(exclusions) or {r["id"] for r in records} & {r["id"] for r in exclusions}:
        raise ValueError("Duplicate or inconsistent excluded record IDs")
    if len(records) + len(exclusions) != manifest["original_labeled_rows"] or len(records) != manifest["retained_rows"]:
        raise ValueError("Input coverage counts disagree")
    seen_doc, seen_component, seen_pair, doc_components = {}, {}, set(), {}
    for record in records:
        source = record["source"]
        if source not in {"train", "dev"} or record["id"] != f"{source}:{record['row_index']:06d}":
            raise ValueError("Record identity or source is invalid")
        if record["split"] not in SPLIT_NAMES or record["label"] not in (0, 1):
            raise ValueError("Invalid split or binary label")
        if record["unordered_pair_group"] in seen_pair:
            raise ValueError("Duplicate unordered pair in retained records")
        seen_pair.add(record["unordered_pair_group"])
        for doc in (record["document_group_1"], record["document_group_2"]):
            if doc in seen_doc and seen_doc[doc] != record["split"]:
                raise ValueError("Shared document crosses split boundaries")
            seen_doc[doc] = record["split"]
            if doc in doc_components and doc_components[doc] != record["group_id"]:
                raise ValueError("Shared document is assigned to different components")
            doc_components[doc] = record["group_id"]
        component = record["group_id"]
        if component in seen_component and seen_component[component] != record["split"]:
            raise ValueError("Connected component crosses split boundaries")
        seen_component[component] = record["split"]
    for split in SPLIT_NAMES:
        labels = Counter(r["label"] for r in records if r["split"] == split)
        if set(labels) != {0, 1} or sum(labels.values()) != manifest["counts"][split]:
            raise ValueError("Every split needs both classes with matching counts")
        if {str(label): labels[label] for label in (0, 1)} != manifest["class_counts"][split]:
            raise ValueError("Class counts disagree")
        if len({r["group_id"] for r in records if r["split"] == split}) != manifest["component_counts"][split]:
            raise ValueError("Connected component counts disagree")
    if verify_sources:
        if repo_root is None:
            raise ValueError("Source root is required")
        for name, source in manifest["source_csvs"].items():
            if name not in {"train", "dev"} or source["filename"] != f"{name}.csv":
                raise ValueError("Unsafe CSV filename")
            path = Path(repo_root) / "data" / source["filename"]
            if not path.is_file() or path.stat().st_size != source["bytes"] or file_sha256(path) != source["sha256"]:
                raise ValueError("Source CSV checksum mismatch")


def load_manifest(path, repo_root=None, *, verify_sources=False):
    manifest = read_manifest(path)
    validate_manifest(manifest, repo_root, verify_sources=verify_sources)
    return manifest


def save_manifest(manifest, path):
    validate_manifest(manifest)
    path = Path(path)
    if path.exists():
        if read_manifest(path) != manifest:
            raise FileExistsError("Preserve the immutable existing split manifest")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    return path


def load_records(repo_root, manifest, split):
    if split not in SPLIT_NAMES:
        raise ValueError("Unknown split")
    manifest = read_manifest(manifest)
    validate_manifest(manifest, repo_root, verify_sources=True)
    originals = {r["id"]: r for name in ("train", "dev") for r in _read_csv(Path(repo_root) / "data" / f"{name}.csv", name)}
    if {r["id"] for r in manifest["records"]} | {r["id"] for r in manifest["excluded_rows"]} != set(originals):
        raise ValueError("Manifest does not account for every original labeled record")
    normalized_seen = {}
    # Validate actual executed cleanup identities across ALL split boundaries,
    # even when this caller requests only one split. No model is run here.
    for selected in manifest["records"]:
        row = originals.get(selected["id"])
        if row is None or row["label"] != selected["label"] or any(text_sha256(row[key]) != selected[key + "_sha256"] for key in ("text_1", "text_2")):
            raise ValueError("Manifest record differs from its source CSV")
        for key in ("text_1", "text_2"):
            for mode, text in identity_views(row[key]).items():
                if not text:
                    raise ValueError("Retained model input becomes empty")
                identity = (mode, text_sha256(text))
                if identity in normalized_seen and normalized_seen[identity] != selected["split"]:
                    raise ValueError("Executed normalized text crosses split boundaries")
                normalized_seen[identity] = selected["split"]
    for excluded in manifest["excluded_rows"]:
        if originals[excluded["id"]]["label"] != excluded["label"]:
            raise ValueError("Excluded record label differs from original CSV")
    chosen = [r for r in manifest["records"] if r["split"] == split]
    records = []
    for selected in chosen:
        row = originals.get(selected["id"])
        if row is None or row["label"] != selected["label"] or any(text_sha256(row[key]) != selected[key + "_sha256"] for key in ("text_1", "text_2")):
            raise ValueError("Manifest record differs from its source CSV")
        records.append({"id": row["id"], "text_1": row["text_1"], "text_2": row["text_2"],
                        "label": row["label"], "group_id": selected["group_id"]})
    return records


def audit_tokenized_input_overlap(tokenizer, records_by_split, *, max_length=512, batch_size=64):
    """Inspect actual truncated input identity without model calls or label use.

    Padding is excluded because the runtime masks it. Tokenizer defaults and
    longest-first pair truncation match TransformerRunner. AB and BA are both
    audited, including possible cross-orientation matches across splits.
    """
    if not 8 <= max_length <= 512 or batch_size < 1 or set(records_by_split) - set(SPLIT_NAMES):
        raise ValueError("Invalid tokenizer diagnostic scope or dimensions")
    owners = defaultdict(lambda: defaultdict(set))
    count = 0
    for split, records in records_by_split.items():
        for orientation in ("ab", "ba"):
            first, second = ("text_1", "text_2") if orientation == "ab" else ("text_2", "text_1")
            for offset in range(0, len(records), batch_size):
                batch = records[offset:offset + batch_size]
                encoded = tokenizer([normalize_roberta_text(r[first]) for r in batch],
                                    [normalize_roberta_text(r[second]) for r in batch],
                                    truncation="longest_first", padding=False, max_length=max_length)
                if "input_ids" not in encoded or any(len(value) != len(batch) for value in encoded.values()):
                    raise ValueError("Tokenizer returned inconsistent batch fields")
                for index, record in enumerate(batch):
                    values = {key: [int(v) for v in value[index]] for key, value in encoded.items()}
                    fingerprint = canonical_sha256(values)
                    owners[fingerprint][split].add(record["id"])
                    count += 1
    shared = {key: value for key, value in owners.items() if len(value) > 1}
    affected = {split: sorted({record for values in shared.values() for record in values.get(split, [])}) for split in records_by_split}
    train_holdout = {key: value for key, value in shared.items() if "train" in value and "locked_holdout" in value}
    return {"schema_version": 1, "max_length": max_length, "pair_truncation": "longest_first",
            "orientations": ["ab", "ba"], "encoded_record_orientations": count,
            "unique_encoded_input_fingerprints": len(owners), "cross_split_shared_fingerprints": len(shared),
            "train_locked_holdout_shared_fingerprints": len(train_holdout),
            "affected_record_ids": affected,
            "shared_fingerprints": [{"sha256": key, "splits": {split: sorted(ids) for split, ids in sorted(values.items())}}
                                    for key, values in sorted(shared.items())],
            "labels_used": False, "model_inference_performed": False,
            "scope": "Exact actual tokenizer input identity only; no near-duplicate or unseen-author claim. No splitting, filtering or configuration reselection is performed."}
