"""CPU-default transformer inference and verified local model inventories."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_artifact_inventory(model_dir):
    root = Path(model_dir).resolve(strict=True)
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("Model artifacts cannot contain symbolic links")
        if path.is_file() and path.name != "artifact-inventory.json":
            files.append({"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size,
                          "sha256": sha256_file(path)})
    document = {"schema_version": 1, "files": files}
    (root / "artifact-inventory.json").write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return document


def verify_artifact_inventory(model_dir, required=()):
    root = Path(model_dir).resolve(strict=True)
    inventory_path = root / "artifact-inventory.json"
    if inventory_path.is_symlink() or not inventory_path.is_file():
        raise ValueError("A verified artifact-inventory.json is required before loading model artifacts")
    document = json.loads(inventory_path.read_text(encoding="utf-8"))
    if document.get("schema_version") != 1 or not isinstance(document.get("files"), list):
        raise ValueError("Unsupported artifact inventory")
    names = set()
    for entry in document["files"]:
        name = entry["path"]
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or "\\" in name or ":" in name or name in names:
            raise ValueError("Unsafe or duplicated model artifact path")
        path = root / relative
        if any(parent.is_symlink() for parent in (path, *path.parents) if parent.is_relative_to(root)):
            raise ValueError("Symbolic model artifacts are forbidden")
        if not path.is_file() or path.stat().st_size != entry["bytes"] or sha256_file(path) != entry["sha256"]:
            raise ValueError("Model artifact size or SHA256 mismatch: " + name)
        names.add(name)
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() and p != inventory_path}
    if names != actual or not set(required).issubset(names):
        raise ValueError("Model artifact inventory is incomplete or contains unexpected files")
    return document


class TransformerRunner:
    def __init__(self, model_dir, device="cpu", *, verify=True):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        from transformers.utils import logging as transformer_logging
        transformer_logging.disable_progress_bar()
        self.model_dir = Path(model_dir).resolve(strict=True)
        if device not in ("cpu", "cuda", "cuda:0"):
            raise ValueError("Device must be cpu or cuda")
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        self.inventory = verify_artifact_inventory(self.model_dir, ("config.json", "training-config.json")) if verify else None
        self.metadata = json.loads((self.model_dir / "training-config.json").read_text(encoding="utf-8"))
        self.max_length = int(self.metadata["max_length"])
        if not 8 <= self.max_length <= 512:
            raise ValueError("Unsupported maximum input length")
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, local_files_only=True, trust_remote_code=False)
        if self.metadata.get('format')=='transformer-dynamic-int8':
            if self.device.type!='cpu':raise ValueError('Dynamic INT8 inference is CPU-only')
            from transformers import AutoConfig
            config=AutoConfig.from_pretrained(self.model_dir,local_files_only=True,trust_remote_code=False)
            self.model=AutoModelForSequenceClassification.from_config(config,attn_implementation='eager')
            self.model=torch.ao.quantization.quantize_dynamic(self.model,{torch.nn.Linear},dtype=torch.qint8)
            state=torch.load(self.model_dir/'model-int8.pt',map_location='cpu',weights_only=True)
            self.model.load_state_dict(state,strict=True)
        else:
            if not (self.model_dir/'model.safetensors').is_file():raise ValueError('F32 model weights are missing')
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_dir, local_files_only=True, trust_remote_code=False, attn_implementation="eager")
        if self.model.config.num_labels != 2:
            raise ValueError("Authorship models must have exactly two logits")
        self.model.to(self.device).eval()

    def predict_logits(self, records, batch_size=16, orientation="ab"):
        import numpy as np
        import torch
        from .normalization import normalize_roberta_text
        if orientation not in ("ab", "ba") or not 1 <= int(batch_size) <= 256:
            raise ValueError("Invalid prediction orientation or batch size")
        result = []
        with torch.inference_mode():
            for offset in range(0, len(records), batch_size):
                batch = records[offset:offset + batch_size]
                first, second = ("text_1", "text_2") if orientation == "ab" else ("text_2", "text_1")
                inputs = self.tokenizer([normalize_roberta_text(r[first]) for r in batch],
                                        [normalize_roberta_text(r[second]) for r in batch],
                                        truncation="longest_first", padding=True, max_length=self.max_length,
                                        return_tensors="pt")
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
                logits = self.model(**inputs).logits.float().cpu().numpy()
                if not np.isfinite(logits).all():
                    raise RuntimeError("Model produced nonfinite logits")
                result.append(logits)
        return np.concatenate(result, axis=0) if result else np.empty((0, 2), dtype=np.float32)


def load_transformer(model_dir, device="cpu"):
    return TransformerRunner(model_dir, device=device)


def export_dynamic_int8(model_dir, output_dir):
    """Create a CPU candidate. Validation, speed gates, and adoption are external."""
    import torch
    import shutil
    source=Path(model_dir).resolve(strict=True)
    verify_artifact_inventory(source,('model.safetensors','config.json','training-config.json'))
    target=Path(output_dir)
    if target.exists() and any(target.iterdir()):raise ValueError('INT8 output directory must be empty')
    target.mkdir(parents=True,exist_ok=True)
    runner=TransformerRunner(source,device='cpu')
    model=torch.ao.quantization.quantize_dynamic(runner.model,{torch.nn.Linear},dtype=torch.qint8)
    torch.save(model.state_dict(),target/'model-int8.pt')
    for name in ('config.json','tokenizer.json','tokenizer_config.json','vocab.json','merges.txt','special_tokens_map.json','added_tokens.json'):
        if (source/name).is_file():shutil.copyfile(source/name,target/name)
    metadata={**runner.metadata,'format':'transformer-dynamic-int8','source_model_sha256':sha256_file(source/'model.safetensors'),
              'quantized_operations':'Linear dynamic qint8; embeddings remain FP32','quantization_device':'cpu',
              'adoption_requires_selection_evaluation':True}
    (target/'training-config.json').write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf8')
    report={'format':metadata['format'],'source_model_sha256':metadata['source_model_sha256'],
            'quantized_model_sha256':sha256_file(target/'model-int8.pt'),'quantized_model_bytes':(target/'model-int8.pt').stat().st_size,
            'source_model_bytes':(source/'model.safetensors').stat().st_size,'final_holdout_used':False}
    (target/'export-report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    write_artifact_inventory(target)
    return report
