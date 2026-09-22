# Authorship Verification

[한국어](./README.ko.md) · [Reproduction guide](./docs/reproduction.md) · [Stylometric model card](./model-cards/stylometric-ensemble.md) · [RoBERTa model card](./model-cards/roberta-asymmetric-loss.md) · [Project poster](./docs/authorship-verification-poster.pdf)

An English authorship-verification project that predicts whether two text passages were written by the same person. It compares an interpretable stylometric ensemble with a fine-tuned RoBERTa-Large model, showing the trade-off between efficient inference and stronger predictive performance.

## Highlights

- **0.8348 Macro F1** with RoBERTa-Large and Asymmetric Loss.
- **0.7702 Macro F1** with a lightweight XGBoost-LightGBM ensemble that can run on CPU.
- End-to-end notebooks for preprocessing, training, threshold optimization, evaluation, batch inference, and interactive prediction.
- Detailed model cards covering architecture, hyperparameters, compute requirements, limitations, and responsible use.

## Results

Both models were evaluated on the same held-out set of 5,993 balanced text pairs.

| Model | Approach | Macro F1 | Accuracy | Inference profile |
|---|---|---:|---:|---|
| Stylometric Ensemble | Word, character, and function-word TF-IDF; 24 style features; XGBoost + LightGBM | 0.7702 | 0.77 | CPU-friendly, under 60 MB of model artifacts |
| RoBERTa + ASL | RoBERTa-Large pair classification with asymmetric focal penalties | **0.8348** | **0.83** | GPU recommended, about 1.32 GB of weights |

The reported values come from the saved evaluation runs documented in the model cards.

## How it works

### Stylometric ensemble

The first pipeline normalizes obvious metadata, extracts word and character n-grams, models function-word usage, and computes 24 handcrafted style signals such as sentence length, punctuation ratios, lexical diversity, and suffix patterns. Pairwise differences, products, ratios, and cosine distances are passed to XGBoost and LightGBM. Their probabilities are combined with tuned weights and a tuned decision threshold.

### RoBERTa-Large with Asymmetric Loss

The second pipeline cleans email headers, addresses, URLs, and redundant whitespace before tokenizing each pair up to 512 tokens. RoBERTa-Large is fine-tuned with Asymmetric Loss (`gamma_neg=2`, `gamma_pos=1`) so training places more emphasis on difficult negative pairs. A threshold sweep on the development set selected `0.51` for the final classifier.

## Dataset

| Split | Pairs | Labels | Purpose |
|---|---:|---|---|
| `train.csv` | 27,643 | Yes | Model fitting |
| `dev.csv` | 5,993 | Yes | Evaluation and threshold tuning |
| `test.csv` | 5,985 | No | Batch prediction |
| `live.csv` | 20 | No | Small inference examples |

Each row contains `text_1` and `text_2`; labelled splits also contain a binary `label` where `1` means same author and `0` means different authors.

## Repository structure

```text
.
├── data/                         # Train, development, test, and sample pairs
├── notebooks/
│   ├── stylometric-ensemble/     # Training and inference notebooks
│   └── roberta-asymmetric-loss/  # Training and inference notebooks
├── model-cards/                  # Detailed model documentation
├── models/                       # Downloaded weights (ignored by Git)
├── results/                      # Saved test predictions
├── docs/                         # Reproduction guide and project poster
├── requirements.txt
├── README.md                     # English overview
└── README.ko.md                  # Korean overview
```

## Quick start

Python 3.12 is recommended.

```bash
git clone https://github.com/Dongmyung378/NLU-Authorship-Verification.git
cd NLU-Authorship-Verification
python -m venv .venv

# PowerShell
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
jupyter lab
```

Download the trained weights and place them under `models/stylometric-ensemble/` or `models/roberta-asymmetric-loss/`, then run the matching demo notebook. Download links, expected files, CUDA notes, and path overrides are listed in the [reproduction guide](./docs/reproduction.md).

## Limitations and responsible use

- The models are optimized for English and may not generalize to multilingual, code-switched, very short, or out-of-domain text.
- RoBERTa truncates combined inputs beyond 512 tokens; the stylometric pipeline ignores unseen TF-IDF vocabulary.
- Performance figures describe one held-out dataset and should not be treated as universal benchmarks.
- Authorship signals can be sensitive. Do not use predictions as sole evidence for identity, attribution, punitive decisions, or deanonymization.

## Contributors

- Dongmyung Park
- Juho Kim
