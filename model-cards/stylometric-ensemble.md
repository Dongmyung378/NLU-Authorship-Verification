---
language: en
license: cc-by-4.0
tags:
  - text-classification
  - authorship-verification
  - stylometry
  - xgboost
  - lightgbm
repo: https://github.com/Dongmyung378/NLU-Authorship-Verification
---

# Stylometric XGBoost-LightGBM Ensemble

[한국어](./stylometric-ensemble.ko.md) · [Project overview](../README.md)

## Model summary

This supervised binary classifier predicts whether two English passages share an author. It combines word-, character-, and function-word TF-IDF signals with handcrafted stylometric features, then blends XGBoost and LightGBM probabilities using tuned ensemble weights and a tuned decision threshold.

- **Developed by:** Dongmyung Park and Juho Kim
- **Language:** English
- **Task:** Pairwise authorship verification
- **Architecture:** XGBoost + LightGBM weighted ensemble
- **Training approach:** From scratch on labelled text pairs

## Input representation

The pipeline uses:

- Word TF-IDF with 1–2 grams and up to 7,000 features.
- Character TF-IDF with 2–4 grams and up to 7,000 features.
- Function-word TF-IDF built from a fixed English vocabulary.
- 24 style features covering length statistics, punctuation, capitalization, lexical diversity, and suffix patterns.
- Pairwise absolute differences, element-wise products, ratios, means, and cosine distances.

Preprocessing removes emails, URLs, dates, phone numbers, long repeated-character sequences, repeated punctuation, and redundant whitespace.

## Training data

The model used 27,643 labelled English text pairs drawn from multiple domains. No external training data was added. The repository contains the train, development, and unlabelled test splits under [`data/`](../data/).

## Hyperparameters

### XGBoost

| Parameter | Value |
|---|---:|
| Estimators | 7,000; early stopping at iteration 6,213 |
| Learning rate | 0.01 |
| Max depth | 7 |
| Subsample | 0.8 |
| Column sample by tree | 0.8 |
| Minimum child weight | 2 |
| Max bins | 128 |

### LightGBM

| Parameter | Value |
|---|---:|
| Estimators | 2,000; early stopping at iteration 1,700 |
| Learning rate | 0.01 |
| Leaves | 127 |
| Subsample | 0.8 |
| Column sample by tree | 0.8 |
| Max bins | 127 |

### Ensemble

| Setting | Value |
|---|---:|
| XGBoost weight | 0.26 |
| LightGBM weight | 0.74 |
| Decision threshold | 0.43 |

## Evaluation

Evaluation used 5,993 development pairs: 2,937 different-author pairs and 3,056 same-author pairs.

| Metric | Score |
|---|---:|
| Macro F1 | **0.7702** |
| Same-author F1 | 0.7813 |
| Accuracy | 0.77 |
| Macro precision | 0.77 |
| Macro recall | 0.77 |

Per-class results:

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| Different author | 0.78 | 0.74 | 0.76 |
| Same author | 0.76 | 0.80 | 0.78 |

## Compute and artifacts

- Training time: approximately 1 hour 10 minutes.
- Training hardware: two NVIDIA T4 GPUs with 16 GB VRAM each, plus a multi-core CPU.
- Inference: standard CPU supported; approximately 2 GB RAM recommended.
- Serialized ensemble, vectorizers, and configuration: under 60 MB total.
- [Download trained artifacts](https://drive.google.com/drive/folders/1mF0bMpzfs3XODpQA2Hd7H_JvXLk5AGgp?usp=sharing).

## Limitations and risks

- The feature set is designed for English; multilingual or code-switched inputs are not supported reliably.
- Style features become less informative on passages shorter than five words.
- TF-IDF ignores terms outside the fitted vocabulary.
- Genre shifts can change stylistic patterns and reduce accuracy.
- Authorship predictions can enable unwanted deanonymization. They should not be treated as conclusive identity evidence.

## Reproduction

See the [training notebook](../notebooks/stylometric-ensemble/training.ipynb), [demo notebook](../notebooks/stylometric-ensemble/demo.ipynb), and [reproduction guide](../docs/reproduction.md).
