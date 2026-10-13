import csv
import json
from pathlib import Path


def write_dataset(root, extras=True):
    root = Path(root)
    (root / "data").mkdir(parents=True, exist_ok=True)
    rows = [{"text_1": f"First passage number {i} contains a sentence and distinctive prose.",
             "text_2": f"Second passage number {i} records another story with its own details.",
             "label": str(i % 2)} for i in range(100)]
    train, dev = rows[:80], rows[80:]
    if extras:
        original = dict(train[0])
        dev.append({"text_1": original["text_2"], "text_2": original["text_1"], "label": "0.0"})
        train.extend([{"text_1": "Conflict first passage.", "text_2": "Conflict other passage.", "label": "0"},
                      {"text_1": "https://example.invalid/only", "text_2": "An actual passage.", "label": "1"},
                      {"text_1": "e\u0301e\u0301e\u0301e\u0301", "text_2": "Unicode partner alpha.", "label": "1"}])
        dev.extend([{"text_1": "Conflict other passage.", "text_2": "Conflict first passage.", "label": "1"},
                    {"text_1": "éé", "text_2": "Unicode partner beta.", "label": "0"}])
    for name, source in (("train", train), ("dev", dev)):
        with (root / "data" / f"{name}.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["text_1", "text_2", "label"])
            writer.writeheader()
            writer.writerows(source)
    return root


def write_source(root):
    root = Path(root)
    (root / "nlu_authorship").mkdir(parents=True, exist_ok=True)
    (root / "nlu_authorship" / "bound.py").write_text("VALUE = 1\n", encoding="utf-8")
    (root / "requirements.txt").write_text("numpy==1.26.4\n", encoding="utf-8")
    return root


def write_artifact(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(json.dumps({"num_labels": 2}), encoding="utf-8")
    (root / "weights.bin").write_bytes(b"saved-test-weights")
    return root
