"""Artifact path resolution and Google Drive downloader for ML and DL checkpoints."""
from __future__ import annotations

import re
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DRIVE_FOLDER_URL = "https://drive.google.com/drive/folders/1ij9ZwlLtimS26bAE9B_D5zL3raAKJody"

ARTIFACTS = {
    "ml": {
        "filename": "model_01_stylometric_ml.joblib",
        "drive_id": "1q9T8zLYmnhLqPWT9He8wiqIjA0kvCR1M",
        "expected_bytes": 22682680,
        "search_paths": (
            PROJECT_ROOT / "outputs" / "01_stylometric_ml" / "model_01_stylometric_ml.joblib",
            PROJECT_ROOT / "artifacts" / "models" / "model_01_stylometric_ml.joblib",
        ),
        "default_dest": PROJECT_ROOT / "outputs" / "01_stylometric_ml" / "model_01_stylometric_ml.joblib",
    },
    "dl": {
        "filename": "model_03_deberta_v3_large.pt",
        "drive_id": "1r19ezaf0Q_nn3qVS43m4nDhAs5jfGkve",
        "expected_bytes": 902056343,
        "search_paths": (
            PROJECT_ROOT / "outputs" / "03_deberta_v3_large" / "model_03_deberta_v3_large.pt",
            PROJECT_ROOT / "artifacts" / "models" / "model_03_deberta_v3_large.pt",
        ),
        "default_dest": PROJECT_ROOT / "outputs" / "03_deberta_v3_large" / "model_03_deberta_v3_large.pt",
    },
}

TOKENIZER_SEARCH_PATHS = (
    PROJECT_ROOT / "outputs" / "03_deberta_v3_large" / "tokenizer_deberta_v3_large",
    PROJECT_ROOT / "artifacts" / "models" / "tokenizer_deberta_v3_large",
)


def find_artifact(kind: str) -> Path | None:
    spec = ARTIFACTS[kind]
    for cand in spec["search_paths"]:
        if cand.is_file() and cand.stat().st_size > 0:
            return cand
    return None


def find_local_tokenizer() -> Path | None:
    for cand in TOKENIZER_SEARCH_PATHS:
        if (cand / "config.json").is_file() and (cand / "tokenizer.json").is_file():
            return cand
    return None


def download_from_drive(kind: str, dest: Path | None = None) -> Path:
    """Download a public model artifact from Google Drive, handling large-file confirm tokens."""
    spec = ARTIFACTS[kind]
    target = dest or spec["default_dest"]
    target.parent.mkdir(parents=True, exist_ok=True)
    file_id = spec["drive_id"]
    base_url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"

    req = urllib.request.Request(base_url, headers={"User-Agent": "Mozilla/5.0"})
    tmp = target.with_suffix(target.suffix + ".part")
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            ctype = resp.headers.get("Content-Type", "")
            if "text/html" in ctype.lower():
                html = resp.read().decode("utf-8", errors="ignore")
                m_uuid = re.search(r'name="uuid"\s+value="([^"]+)"', html)
                if m_uuid:
                    url2 = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t&uuid={m_uuid.group(1)}"
                    with urllib.request.urlopen(urllib.request.Request(url2, headers={"User-Agent": "Mozilla/5.0"}), timeout=300) as r2, open(tmp, "wb") as f:
                        while chunk := r2.read(1024 * 1024):
                            f.write(chunk)
                else:
                    raise RuntimeError(f"Unexpected HTML response when downloading {spec['filename']}. Download manually from {DRIVE_FOLDER_URL}")
            else:
                with open(tmp, "wb") as f:
                    while chunk := resp.read(1024 * 1024):
                        f.write(chunk)
        tmp.replace(target)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    return target
