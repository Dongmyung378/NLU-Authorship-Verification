---
language: en
license: cc-by-4.0
tags:
  - text-classification
  - authorship-verification
  - roberta-large
  - asymmetric-loss
  - pytorch
repo: https://github.com/Dongmyung378/NLU-Authorship-Verification
---

# RoBERTa-Large with Asymmetric Loss

[한국어](./roberta-asymmetric-loss.ko.md) · [Project overview](../README.md)

## Model summary

This supervised binary classifier predicts whether two English passages share an author. It fine-tunes `roberta-large` with a sequence-classification head and Asymmetric Loss, placing extra emphasis on difficult negative pairs with subtle stylistic differences.

- **Developed by:** Dongmyung Park and Juho Kim
- **Language:** English
- **Task:** Pairwise authorship verification
- **Architecture:** RoBERTa-Large with a binary classification head
- **Base model:** [`FacebookAI/roberta-large`](https://huggingface.co/FacebookAI/roberta-large)

## Input representation

Email headers, email addresses, URLs, and redundant whitespace are removed before tokenization. The two passages are encoded as a standard RoBERTa pair and truncated with `longest_first` to a combined maximum of 512 tokens.

The model replaces cross-entropy with Asymmetric Loss using `gamma_neg=2` and `gamma_pos=1`. Easy examples are down-weighted, with a stronger focal penalty on the negative class. A development-set threshold sweep selected `0.51` for inference.

## Training data

The model used 27,643 labelled English text pairs drawn from multiple domains. No external training data was added. The repository contains the train, development, and unlabelled test splits under [`data/`](../data/).

## Hyperparameters

| Parameter | Value |
|---|---:|
| Learning rate | 5e-6 |
| Training batch size | 32 |
| Evaluation batch size | 32 |
| Maximum sequence length | 512 |
| Epochs | 14; early stopping at epoch 13 |
| Warmup ratio | 0.1 |
| Weight decay | 0.01 |
| Loss | Asymmetric Loss (`gamma_neg=2`, `gamma_pos=1`) |
| Decision threshold | 0.51 |

## Evaluation

Evaluation used 5,993 development pairs: 2,937 different-author pairs and 3,056 same-author pairs.

| Metric | Score |
|---|---:|
| Macro F1 | **0.8348** |
| Same-author F1 | 0.8365 |
| Accuracy | 0.83 |
| Macro precision | 0.83 |
| Macro recall | 0.83 |

Per-class results:

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| Different author | 0.83 | 0.84 | 0.83 |
| Same author | 0.84 | 0.83 | 0.84 |

## Compute and artifacts

- Training time: approximately 6 hours 11 minutes.
- Training hardware: one NVIDIA A100 40 GB GPU.
- Model parameters: approximately 355 million.
- Model weights: approximately 1.32 GB.
- Inference: 16 GB system RAM and a CUDA-capable GPU are recommended; CPU inference is possible but slow.
- [Download trained artifacts](https://drive.google.com/drive/folders/1Ty9-EpOLAgzvTWPuqrHBTbehmZZuDs3V?usp=sharing).

## Limitations and risks

- Combined inputs longer than 512 tokens are truncated and may lose useful evidence.
- The base model and fine-tuning data are English-focused; multilingual, code-switched, or dialect-heavy inputs may be unreliable.
- The model is costly to train and comparatively heavy to deploy.
- Genre and domain shifts can reduce accuracy.
- Authorship predictions can enable unwanted deanonymization. They should not be treated as conclusive identity evidence.

## Reproduction

See the [training notebook](../notebooks/roberta-asymmetric-loss/training.ipynb), [demo notebook](../notebooks/roberta-asymmetric-loss/demo.ipynb), and [reproduction guide](../docs/reproduction.md).
