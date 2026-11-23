# Reproduction & Live Demo Guide

[한국어](./reproduction.ko.md) | [README](../README.md)

## 1. Local Interactive Web Demo & CLI

```powershell
# Install Python dependencies (Python 3.10+ recommended)
pip install -r requirements.txt

# Download public model weights from Google Drive
python verify_author.py download --model ml    # 21.63 MB (CPU)
python verify_author.py download --model dl    # 860.27 MB (GPU recommended, CPU supported)

# Check installed artifacts
python verify_author.py status

# Launch MIT Red interactive web interface (http://127.0.0.1:8000)
.\launch-nlu.bat demo
# Linux / macOS:
sh ./launch-nlu.sh demo

# CLI single-pair verification (ml, dl, or both)
python verify_author.py predict --model both --check-symmetry --text-file-1 a.txt --text-file-2 b.txt

# Run automated unit, API, symmetry, and locked-test parity checks
pytest -v tests/test_inference_and_demo.py
```

## 2. Full Training Reproduction in Google Colab

Each track has a clean, self contained single-pass training notebook under `notebooks/colab/` (and mirrored directly on Google Drive with Colab MIME type):

1. Download [benchmark_1024.zip (61.2 MB)](https://drive.google.com/file/d/1Y9xfMRskEUZPYw9f5jdiM6ws9L-Cwdr8/view) from the [public Google Drive folder](https://drive.google.com/drive/folders/1ij9ZwlLtimS26bAE9B_D5zL3raAKJody).
2. Open the target notebook in Google Colab and upload `benchmark_1024.zip`:
   - **Track 01 (CPU 8+ vCPU, 32 GB RAM)**: [train_01_stylometric_ml.ipynb](https://colab.research.google.com/drive/1jAuZKPe7mcuL9jB9SAIQmGr-jQ-CdjtJ) → exports `model_01_stylometric_ml.joblib` and `results_01_stylometric_ml.zip`.
   - **Track 02 (L4 24GB / A100)**: [train_02_deberta_v3_base.ipynb](https://colab.research.google.com/drive/1U6Fv97IcC061XdJlVYWM79D9YSOsD9Bk) → exports `results_02_deberta_v3_base.zip`.
   - **Track 03 (L4 24GB / A100 / H100)**: [train_03_deberta_v3_large.ipynb](https://colab.research.google.com/drive/1ROS0vdfbpZupVoM38Dz41eDB1tCAGjWL) → exports `model_03_deberta_v3_large.pt` and `results_03_deberta_v3_large.zip`. Uploading `results_01_stylometric_ml.zip` and `results_02_deberta_v3_base.zip` alongside `benchmark_1024.zip` automatically executes the 3 way Exact McNemar & Bootstrap comparison.
3. Run all cells top-to-bottom. All default flags in `CONFIG` match the final winning recipes.
