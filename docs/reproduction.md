# Reproduction Guide

[한국어](./reproduction.ko.md) · [Project overview](../README.md)

This guide covers local setup, model artifact placement, evaluation, batch inference, and optional retraining for both authorship-verification pipelines.

## 1. Environment

The inference notebooks were validated with Python 3.12.7. Create an isolated environment from the repository root:

```bash
python -m venv .venv

# PowerShell
.\.venv\Scripts\Activate.ps1

# macOS or Linux
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
jupyter lab
```

The pinned `torch` package works for CPU inference. For NVIDIA acceleration, install the PyTorch build that matches the CUDA runtime on your machine before installing the remaining dependencies. Exact training-library versions are recorded in each [model card](../model-cards/).

## 2. Download model artifacts

The trained files are hosted separately because the RoBERTa checkpoint is about 1.32 GB.

### Stylometric ensemble

[Download the XGBoost, LightGBM, and vectorizer artifacts](https://drive.google.com/drive/folders/1mF0bMpzfs3XODpQA2Hd7H_JvXLk5AGgp?usp=sharing), then place them as follows:

```text
models/stylometric-ensemble/
├── ensemble_config.json
├── lgb_model.pkl
├── tfidf_char.pkl
├── tfidf_fw.pkl
├── tfidf_word.pkl
└── xgb_model.json
```

### RoBERTa-Large

[Download the fine-tuned RoBERTa artifacts](https://drive.google.com/drive/folders/1Ty9-EpOLAgzvTWPuqrHBTbehmZZuDs3V?usp=sharing), then place them as follows:

```text
models/roberta-asymmetric-loss/
├── best_threshold.json
├── config.json
├── model.safetensors
├── tokenizer.json
└── tokenizer_config.json
```

Tokenizer folders may contain additional vocabulary files. Keep all downloaded files together in the same directory.

## 3. Data layout

The notebooks expect the repository datasets under `data/`:

```text
data/
├── train.csv
├── dev.csv
├── test.csv
└── live.csv
```

Required columns:

- `text_1`, `text_2`: the input passages.
- `label`: present in training and development data; `1` indicates the same author and `0` indicates different authors.

## 4. Run inference

Start Jupyter from the repository root and open one of these notebooks:

- [`notebooks/stylometric-ensemble/demo.ipynb`](../notebooks/stylometric-ensemble/demo.ipynb)
- [`notebooks/roberta-asymmetric-loss/demo.ipynb`](../notebooks/roberta-asymmetric-loss/demo.ipynb)

Run the cells in order. Each notebook evaluates the labelled development split, generates predictions for `test.csv`, writes them to `results/`, and exposes a single-pair prediction helper.

The stylometric pipeline supports CPU inference. A CUDA-capable GPU is strongly recommended for practical RoBERTa-Large inference.

## 5. Retrain a model

Training notebooks are available at:

- [`notebooks/stylometric-ensemble/training.ipynb`](../notebooks/stylometric-ensemble/training.ipynb)
- [`notebooks/roberta-asymmetric-loss/training.ipynb`](../notebooks/roberta-asymmetric-loss/training.ipynb)

The default output folders are:

- `models/stylometric-ensemble/`
- `models/roberta-asymmetric-loss/`

Training is substantially more demanding than inference. The recorded runs used two NVIDIA T4 GPUs for the boosting ensemble and one NVIDIA A100 40 GB GPU for RoBERTa-Large. Adjust batch size and device settings for the available hardware.

## 6. Path overrides

The notebooks locate the repository root automatically. You can override paths with environment variables when running on hosted notebook services or custom storage:

| Variable | Purpose |
|---|---|
| `AV_PROJECT_ROOT` | Repository root |
| `AV_DATA_DIR` | Dataset directory |
| `AV_MODEL_DIR` | Model input or output directory |
| `AV_RESULTS_DIR` | Prediction output directory |

Restart the Jupyter kernel after changing installed packages or environment variables.

## 7. Expected outputs

Batch inference produces one-column CSV files with the header `prediction`:

```text
results/
├── stylometric-ensemble.csv
└── roberta-asymmetric-loss.csv
```

Each file contains 5,985 predictions aligned row-for-row with `data/test.csv`.
