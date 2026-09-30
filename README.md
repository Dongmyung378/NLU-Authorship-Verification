# Authorship Verification

**A joint NLP project by Dongmyung Park and Juho Kim**

[한국어](./README.ko.md) · [Reproduction guide](./docs/reproduction.md) · [Original course poster](./docs/authorship-verification-poster.pdf)

This portfolio project compares a stylometric ensemble and a fine-tuned RoBERTa-Large model on the supplied English same-author/different-author text-pair dataset. The reported metrics describe the supplied development split. The repository includes the data, training and inference notebooks, saved predictions, model documentation, and a terminal demo for exploring model outputs on other text; performance on new authors or domains has not been established.

## Results on the supplied development split

| Model | Method | Dev Macro F1 | Dev accuracy | Practical trade-off |
|---|---|---:|---:|---|
| [Stylometric ensemble](./model-cards/stylometric-ensemble.md) | Word, character, and function-word TF-IDF; 24 style features; XGBoost + LightGBM | 0.7702 | 0.77 | CPU inference; artifacts under 60 MB |
| [RoBERTa-Large + ASL](./model-cards/roberta-asymmetric-loss.md) | Paired-text transformer with Asymmetric Loss | **0.8348** | **0.83** | Higher score on the supplied dev split; about 1.32 GB of weights |

These are scores on the same **5,993-pair labeled development split** (2,937 different-author and 3,056 same-author pairs). A direct comparison of the supplied CSVs found **630 development pairs (10.5%) that exactly duplicate training pairs**. The development split was also used to select decision thresholds and, for RoBERTa, the best checkpoint. These figures describe the recorded development runs; they do not establish performance on independent new authors or domains. The repository's 5,985-pair test split is unlabeled, so no test-set accuracy or F1 is claimed.

## What we built

The **stylometric pipeline** removes identifying metadata, then represents each passage with word and character n-grams, function-word usage, and 24 style measurements such as sentence length, punctuation, and lexical diversity. Pairwise features feed XGBoost and LightGBM; their outputs are blended with tuned weights and a `0.43` decision threshold. It is the smaller option for local CPU use.

The **RoBERTa pipeline** cleans email headers, addresses, and URLs, encodes both passages together up to 512 tokens, and fine-tunes `roberta-large` with Asymmetric Loss (`gamma_neg=2`, `gamma_pos=1`). Its development-set threshold is `0.51`. This approach captures contextual signals at a higher compute and storage cost.

Both pipelines have training notebooks, inference notebooks, and model cards. [`verify_author.py`](./verify_author.py) provides one consistent command-line interface for single-pair predictions.

## Try a prediction

### 1. Set up Python

Python 3.12 is recommended. Create a virtual environment with the Python 3.12 launcher available on your system, then install the terminal-inference dependencies. For example, in Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-inference.txt
```

If `python` is unavailable, use your installed Python launcher to create `.venv`. Once the environment exists, the commands above and below call its interpreter directly. On macOS or Linux, use `.venv/bin/python` in place of `.\.venv\Scripts\python.exe`. For the notebooks and retraining, install the full [`requirements.txt`](./requirements.txt); see the [reproduction guide](./docs/reproduction.md) for setup and CUDA details.

For **RoBERTa on an NVIDIA GPU in Windows**, install the CUDA 12.4 PyTorch build after the requirements above (and after `requirements.txt` if you also need the notebooks):

```powershell
.\.venv\Scripts\python.exe -m pip install --force-reinstall --no-deps torch==2.5.1+cu124 --index-url https://download.pytorch.org/whl/cu124
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No CUDA device')"
```

The check should print `True` for CUDA availability and your GPU name. Choose a wheel compatible with your platform and NVIDIA driver from the [official PyTorch version list](https://pytorch.org/get-started/previous-versions/) if CUDA 12.4 is unsuitable. In **Git Bash on Windows**, use `./.venv/Scripts/python.exe` in these and the demo commands.

### 2. Download one or both models

Trained artifacts are hosted outside GitHub because of their size. The exact download links are:

| Model | Download | Put the files directly in either folder |
|---|---|---|
| Stylometric ensemble | [Google Drive: XGBoost, LightGBM, and vectorizers](https://drive.google.com/drive/folders/1mF0bMpzfs3XODpQA2Hd7H_JvXLk5AGgp?usp=sharing) | `models/stylometric-ensemble/` or `models/Models_ML/` |
| RoBERTa-Large + ASL | [Google Drive: fine-tuned checkpoint and tokenizer](https://drive.google.com/drive/folders/1Ty9-EpOLAgzvTWPuqrHBTbehmZZuDs3V?usp=sharing) | `models/roberta-asymmetric-loss/` or `models/Models_ASL/` |

The downloaded Google Drive folder names `Models_ML` and `Models_ASL` work as-is for the terminal demo. Place them inside `models/`, with the model files immediately inside each folder; avoid an extra nested folder. The canonical folders are used by the notebooks unless `AV_MODEL_DIR` is set. The `models/` directory is ignored by Git, so downloaded weights stay local.

The terminal demo checks for these files:

```text
models/Models_ML/                 # or models/stylometric-ensemble/
├── ensemble_config.json
├── lgb_model.pkl
├── tfidf_char.pkl
├── tfidf_fw.pkl
├── tfidf_word.pkl
└── xgb_model.json

models/Models_ASL/                # or models/roberta-asymmetric-loss/
├── best_threshold.json
├── config.json
├── model.safetensors
├── tokenizer.json
└── tokenizer_config.json
```

Keep any additional tokenizer files supplied with the RoBERTa download in that same folder. You only need the artifacts for the model you choose to run.

### 3. Run the terminal demo

From the repository root, enter two English passages when prompted. The default is the stylometric ensemble:

```powershell
.\.venv\Scripts\python.exe verify_author.py
```

Select RoBERTa on the GPU, or provide both passages on the command line:

```powershell
.\.venv\Scripts\python.exe verify_author.py --model roberta --device cuda
.\.venv\Scripts\python.exe verify_author.py --text-1 "I walked by the river this morning and watched the boats pass." --text-2 "The rain started quietly in the evening, so I closed the windows."
```

In **Windows Git Bash**, the GPU command is:

```bash
./.venv/Scripts/python.exe verify_author.py --model roberta --device cuda
```

The script prints **Same author** or **Different authors**, `Same-author probability`, and the decision threshold. Despite that output label, the number is an uncalibrated model score, not a measured probability that the authors match; a high score on a new passage does not establish accuracy. Use `--model-dir PATH` if the selected artifacts are elsewhere. For RoBERTa, `--device auto` (the default) uses CUDA when available. If it detects an NVIDIA GPU but PyTorch cannot use CUDA, it stops with setup guidance instead of silently running on CPU. `--device cuda` requires CUDA; use `--device cpu` only when you intentionally want slower CPU inference. The stylometric ensemble runs on CPU.

Scores and thresholds are displayed as percentages.

## Project layout

```text
data/                       Labeled train/dev pairs and unlabeled test/live pairs
notebooks/                  Training and demo notebooks for each pipeline
model-cards/                Architecture, hyperparameters, evaluation, limitations
models/                     Downloaded artifacts (Git-ignored)
results/                    Saved predictions for the unlabeled test split
docs/                       Reproduction guide and project poster
verify_author.py            Interactive and argument-based terminal demo
requirements-inference.txt  Dependencies for the terminal demo
requirements.txt            Full notebook and training dependencies
```

The labeled [`train.csv`](./data/train.csv) split has 27,643 pairs. [`dev.csv`](./data/dev.csv) has 5,993 labeled pairs; [`test.csv`](./data/test.csv) and [`live.csv`](./data/live.csv) have 5,985 and 20 unlabeled pairs, respectively. Each pair has `text_1` and `text_2`; `label=1` means same author and `label=0` means different authors. The notebooks remain available in the repository; [`.gitattributes`](./.gitattributes) excludes `.ipynb` files from GitHub's language statistics.

## Explore the work

- [Reproduction guide](./docs/reproduction.md): local setup, artifact placement, inference, and retraining.
- [Stylometric model card](./model-cards/stylometric-ensemble.md) and [RoBERTa model card](./model-cards/roberta-asymmetric-loss.md): methods, hyperparameters, compute, results, and risks.
- [Training and demo notebooks](./notebooks/): end-to-end experiments and batch inference.
- [Original course poster](./docs/authorship-verification-poster.pdf): historical presentation. Its claim about reducing domain-shift overfitting has not been verified on an independent cross-domain evaluation, and its repository link points to an earlier project location. Use the current model cards for the evaluated scope and the download links above for artifacts.

## Limitations

The methods target English text and may be unreliable for very short passages, other languages, code switching, or unfamiliar genres. RoBERTa truncates combined inputs beyond 512 tokens, while the stylometric model cannot represent terms outside its fitted TF-IDF vocabulary. The reported numbers come from a development split containing exact training-pair duplicates and used for model selection; an independent, author- and domain-separated labeled evaluation is needed to measure generalization. The uncalibrated score reflects patterns learned from the supplied training data, not proof of a person's identity; it should not be the sole basis for attribution, sanctions, or deanonymization.

## Authors

Dongmyung Park and Juho Kim developed this project together.
