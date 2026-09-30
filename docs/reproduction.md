# Reproduction guide

[한국어](./reproduction.ko.md) · [Project overview](../README.md)

Run a local same-author prediction with the terminal script, or use the notebooks to evaluate the development split and generate batch predictions. The trained model files are downloaded separately because they are too large for this GitHub repository.

## 1. Set up Python

Use Python 3.12 (the notebooks were checked with 3.12.7). Run these commands from the repository root. Activation is optional; the examples call the virtual environment directly.

**Windows PowerShell**

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-inference.txt
```

If `py` is unavailable, use `python` to create the environment. For Jupyter notebooks or retraining, also install the full dependency list:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

**macOS or Linux**

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-inference.txt
# For Jupyter notebooks or retraining:
.venv/bin/python -m pip install -r requirements.txt
```

The inference requirements cover the terminal script. `requirements.txt` also includes Jupyter and training dependencies.

**Windows with an NVIDIA GPU:** after installing the requirements above, install the CUDA 12.4 build of PyTorch. Run this after `requirements.txt` too if you installed the full notebook dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install --force-reinstall --no-deps torch==2.5.1+cu124 --index-url https://download.pytorch.org/whl/cu124
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No CUDA device')"
```

The verification should show `2.5.1+cu124`, `12.4`, `True`, and your GPU name. If CUDA availability is `False`, check the NVIDIA driver and the installed wheel before running RoBERTa. CUDA 12.4 is an example for a compatible Windows setup; select a build for your platform and driver from the [official PyTorch previous-versions page](https://pytorch.org/get-started/previous-versions/). Installing a Python package named `torch` alone does not guarantee CUDA support. In Windows Git Bash, write the interpreter path as `./.venv/Scripts/python.exe`; the `.\.venv\Scripts\python.exe` spelling above is for PowerShell.

## 2. Download the model artifacts

Download either model or both. Keep each model's files **directly inside** its directory, without an extra nested Google Drive folder. The entire `models/` directory is ignored by Git and is not included in a clone.

| Model | Download | Canonical directory |
|---|---|---|
| Stylometric XGBoost + LightGBM ensemble | [Model and vectorizer files](https://drive.google.com/drive/folders/1mF0bMpzfs3XODpQA2Hd7H_JvXLk5AGgp?usp=sharing) | `models/stylometric-ensemble/` |
| Fine-tuned RoBERTa-Large | [Checkpoint and tokenizer files](https://drive.google.com/drive/folders/1Ty9-EpOLAgzvTWPuqrHBTbehmZZuDs3V?usp=sharing) | `models/roberta-asymmetric-loss/` |

Expected minimum contents:

```text
models/
├── stylometric-ensemble/
│   ├── ensemble_config.json
│   ├── lgb_model.pkl
│   ├── tfidf_char.pkl
│   ├── tfidf_fw.pkl
│   ├── tfidf_word.pkl
│   └── xgb_model.json
└── roberta-asymmetric-loss/
    ├── best_threshold.json
    ├── config.json
    ├── model.safetensors
    ├── tokenizer.json
    └── tokenizer_config.json
```

Keep any additional downloaded tokenizer or training files in the same directory. The RoBERTa weights alone are approximately 1.3 GiB, so leave enough disk space.

The terminal script also accepts the original download folder names `models/Models_ML/` for the ensemble and `models/Models_ASL/` for RoBERTa. It searches the canonical directory first, then the matching original name, and selects a directory only when all required files are present. **The notebooks use only the canonical directory by default.** To run a notebook with an original folder name, either rename the folder to its canonical name or set `AV_MODEL_DIR` to that exact folder before starting Jupyter (section 5).

## 3. Run a terminal prediction

From the repository root, start the interactive demo. It prompts for two English passages and uses the CPU-friendly stylometric ensemble by default:

```powershell
.\.venv\Scripts\python.exe verify_author.py
```

On macOS or Linux, replace `.\.venv\Scripts\python.exe` with `.venv/bin/python`. You can select RoBERTa, supply both passages on the command line, or point to a model directory outside `models/`:

```powershell
.\.venv\Scripts\python.exe verify_author.py --model roberta --device cuda
.\.venv\Scripts\python.exe verify_author.py --text-1 "I walked by the river this morning." --text-2 "The rain began quietly that evening."
.\.venv\Scripts\python.exe verify_author.py --model roberta --model-dir "D:\models\Models_ASL" --text-1 "First passage." --text-2 "Second passage."
```

For a Unix path, use `--model-dir /path/to/Models_ASL`. The `--model-dir` value must be the directory containing the files above, not its parent. Both `--text-1` and `--text-2` must be provided together; otherwise the script prompts for both.

For **Git Bash on Windows**, run:

```bash
./.venv/Scripts/python.exe verify_author.py --model roberta --device cuda
```

For RoBERTa, `--device auto` is the default and selects CUDA when PyTorch can use it. If an NVIDIA GPU is detected but PyTorch cannot use CUDA, the command stops with setup guidance instead of silently using CPU. `--device cuda` requires CUDA; `--device cpu` explicitly opts into slower CPU inference. The stylometric model uses CPU.

The output reports `Prediction`, `Same-author probability`, and `Decision threshold`. A score **greater than or equal to** the model's saved threshold yields “Same author”; a lower score yields “Different authors.” This is a model estimate for the supplied pair, not proof of identity or a guaranteed calibrated probability.

## 4. Run the notebooks and inspect results

Install `requirements.txt` as in section 1, then start Jupyter from the repository root:

```powershell
.\.venv\Scripts\jupyter.exe lab
```

On macOS or Linux, run `.venv/bin/jupyter lab`. Open either [stylometric demo](../notebooks/stylometric-ensemble/demo.ipynb) or [RoBERTa demo](../notebooks/roberta-asymmetric-loss/demo.ipynb), then run the cells in order. Each notebook:

1. Loads its model files and evaluates labeled `data/dev.csv`.
2. Predicts the unlabeled `data/test.csv` rows and writes a one-column CSV under `results/`.
3. Defines `predict_author_match` and runs a sample pair. Edit `text_a` and `text_b` in the last cell for your own passages.

The saved files are `results/stylometric-ensemble.csv` and `results/roberta-asymmetric-loss.csv`. Each has a `prediction` column with one 0 or 1 per test row, in the original row order. The current `test.csv` has 5,985 rows. Here `1` means same author and `0` means different authors. The test split has no labels, so its prediction CSV is not a measured test score. Development metrics are reported on `dev.csv`, which was also used to select thresholds; interpret them as development results.

## 5. Data and path overrides

The supplied datasets live in `data/`:

| File | Columns | Use |
|---|---|---|
| `train.csv` | `text_1`, `text_2`, `label` | Training |
| `dev.csv` | `text_1`, `text_2`, `label` | Development evaluation and threshold selection |
| `test.csv` | `text_1`, `text_2` | Unlabeled batch inference |
| `live.csv` | `text_1`, `text_2` | Additional unlabeled pairs |

The notebooks look upward from the working directory for a repository containing `data/`. Set these environment variables before starting Jupyter if your files are elsewhere:

| Variable | Notebook path |
|---|---|
| `AV_PROJECT_ROOT` | Repository root |
| `AV_DATA_DIR` | Directory containing the CSV files |
| `AV_MODEL_DIR` | Directory containing the **selected model's files** |
| `AV_RESULTS_DIR` | Directory for output prediction CSV files |

For example, to keep an unrenamed ensemble download on Windows:

```powershell
$env:AV_MODEL_DIR = (Resolve-Path .\models\Models_ML).Path
.\.venv\Scripts\jupyter.exe lab
```

On macOS or Linux, use `AV_MODEL_DIR="$PWD/models/Models_ML" .venv/bin/jupyter lab`. Change `Models_ML` to `Models_ASL` when running the RoBERTa notebook. The terminal script uses `--model-dir` instead of `AV_MODEL_DIR`. Restart the Jupyter kernel after changing environment variables or installed packages.

## 6. Optional retraining and hardware

The [stylometric training notebook](../notebooks/stylometric-ensemble/training.ipynb) and [RoBERTa training notebook](../notebooks/roberta-asymmetric-loss/training.ipynb) read `train.csv` and `dev.csv`. They save artifacts to the canonical model directories by default; `AV_MODEL_DIR` can change the output directory. The RoBERTa notebook loads the base `roberta-large` model, which may need a separate download if it is not cached.

RoBERTa-Large training needs substantially more memory and compute than inference. The [model cards](../model-cards/) record the original model setup and results.
