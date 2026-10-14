---
language: en
tags:
  - text-classification
  - authorship-verification
  - roberta-large
  - focal-loss
  - pytorch
repo: https://github.com/Dongmyung378/NLU-Authorship-Verification
---

# RoBERTa-Large with custom binary focal (historical name: ASL)

[한국어](./roberta-asymmetric-loss.ko.md) | [Project overview](../README.md)

This card describes the original RoBERTa-Large development artifact. Its scores are historical development results and are not a controlled comparator for the new document-disjoint RoBERTa-Large experiment, which starts from fresh pinned pretrained weights.

## Model summary

This supervised binary classifier was trained to label pairs in the supplied English text dataset as same-author or different-author. It fine-tunes `roberta-large` with a sequence-classification head and a custom binary asymmetric focal loss. Its reported performance is limited to the supplied development split.

- **Developed by:** Dongmyung Park and Juho Kim
- **Language:** English
- **Task:** Pairwise authorship verification
- **Architecture:** RoBERTa-Large with a binary classification head
- **Base model:** [`FacebookAI/roberta-large`](https://huggingface.co/FacebookAI/roberta-large)

## Input representation

Email headers, email addresses, URLs, and redundant whitespace are removed before tokenization. The two passages are encoded as a standard RoBERTa pair and truncated with `longest_first` to a combined maximum of 512 tokens.

The model replaces cross-entropy with the custom loss `-(1-p_y)^gamma_y * log(max(p_y, 1e-8))`, averaged over examples, using a two-class softmax with `gamma_neg=2` and `gamma_pos=1`. It down-weights easy examples more strongly for class 0. The original notebook calls this class `ASLoss`; its implementation is a custom class-dependent binary focal loss, neither official multi-label ASL nor official `ASLSingleLabel`. Both official variants exist. The `1e-8` floor is a numerical safeguard, not ASL's negative-probability shifting mechanism. The [loss terminology comparison](../docs/loss-terminology.md) explains the official methods and how the maintained FP32 log-softmax differs at extreme probabilities. The original notebook names and paths remain historical references. A development-set threshold sweep selected `0.51` for inference. A causal improvement in cross-domain generalization has not been established.

## Training data

The model used the 27,643 labeled English text pairs in the supplied [`train.csv`](../data/train.csv); no additional labeled pairs were added for task-specific training. The labeled train and development CSVs contain passages and pair labels, but no author IDs or domain labels, so an author-separated or domain-separated evaluation cannot be verified from these files.

## Hyperparameters

| Parameter | Value |
|---|---:|
| Learning rate | 5e-6 |
| Training batch size | 32 |
| Evaluation batch size | 32 |
| Maximum sequence length | 512 |
| Epoch ceiling | 14 |
| Saved best checkpoint | Epoch 10, step 8,640 |
| Early stopping patience | 3 evaluation epochs |
| Warmup ratio | 0.1 |
| Weight decay | 0.01 |
| Loss | Custom binary asymmetric focal loss (`gamma_neg=2`, `gamma_pos=1`) |
| Decision threshold | 0.51 |

The downloaded checkpoint's `trainer_state.json` records epoch 10.0, global step 8,640 and best metric 0.8346138164975799. The epoch-13 stopping point stated in an earlier card is not established by this saved checkpoint's state. A selected best checkpoint and the final training stopping point are separate facts.

## Evaluation

Evaluation used 5,993 development pairs: 2,937 different-author pairs and 3,056 same-author pairs. A comparison of the supplied CSVs found 630 development pairs (10.5%) that exactly repeat a text pair in the training split, with matching labels. The development split was also used for checkpoint selection and the decision threshold. These figures are development results and do not establish performance on independent, unseen authors or domains.

| Metric | Score |
|---|---:|
| Macro F1 | **0.8348** |
| Same-author F1 | 0.8365 |
| Accuracy | 0.83 |
| Macro precision | 0.83 |
| Macro recall | 0.83 |

An additional audit reran the saved checkpoint on the same `data/dev.csv` using the notebook's preprocessing and the saved 0.51 threshold. It found:

| Development subset | Pairs | Macro F1 |
|---|---:|---:|
| Combined passage length of 50 words or fewer | 554 | 0.6680 |
| Combined passage length over 100 words | 4,775 | 0.8599 |
| Exact text-pair repeats from training | 630 | 0.9857 |
| Excluding exact text-pair repeats | 5,363 | 0.8170 |

Word counts use whitespace splitting. These are overlapping, post-hoc views of the same development split, not independent tests or estimates for new domains.

Per-class results:

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| Different author | 0.83 | 0.84 | 0.83 |
| Same author | 0.84 | 0.83 | 0.84 |

## Compute and artifacts

- Training time: approximately 6 hours 11 minutes.
- Training hardware: one NVIDIA A100 40 GB GPU.
- Model parameters: approximately 355 million.
- Model weights: 1,421,495,360 bytes, approximately 1.42 GB or 1.32 GiB.
- Inference: 16 GB system RAM and a CUDA-capable GPU with a CUDA-enabled PyTorch build are recommended; CPU inference is possible but slow.
- [Download trained artifacts](https://drive.google.com/drive/folders/1Ty9-EpOLAgzvTWPuqrHBTbehmZZuDs3V?usp=sharing). The downloaded model files are kept outside GitHub; the repository ignores `models/`.

## Run the downloaded model

From the repository root, download all files from the link above and put them directly in `models/Models_ASL/` (the original download folder name) or `models/roberta-asymmetric-loss/`. The directory must contain `best_threshold.json`, `config.json`, `model.safetensors`, `tokenizer.json`, and `tokenizer_config.json`. Keep any additional downloaded tokenizer files together, and avoid an extra nested folder between the chosen directory and these files.

Install the [historical terminal demo dependencies](../requirements-inference.txt). The example below explicitly uses CPU and prompts for two English passages. Original GPU training and demo procedures remain in the notebooks linked below; the current [reproduction guide](../docs/reproduction.md) describes the new frozen CPU release rather than a CUDA setup for this historical artifact:

```bash
python -m pip install -r requirements-inference.txt
python verify_author.py --model roberta --device cpu
```

The default `--device auto` selects CUDA when available. If an NVIDIA GPU is detected but PyTorch cannot use CUDA, it stops with setup guidance; it uses CPU when no NVIDIA GPU is detected. Use `--device cuda` to require GPU inference, or `--device cpu` to force CPU inference. To pass text on the command line, use `python verify_author.py --model roberta --text-1 "First English passage" --text-2 "Second English passage"`. The output includes `Same-author probability`, the loaded model's decision threshold, and the resulting label. Despite the displayed name, this is a raw class-1 softmax score, not a calibrated probability that the passages share an author. Results for arbitrary new authors or domains have not been established. If the files live elsewhere, add `--model-dir PATH` with the directory that directly contains them. The [current reproduction guide](../docs/reproduction.md) covers the new frozen CPU release, whose artifacts and evaluation bindings are separate from this historical demo.

## Limitations and risks

- Combined inputs longer than 512 tokens are truncated and may lose useful evidence.
- Very short passages may be unreliable: the median training passage has about 123 whitespace-separated words, much longer than a short terminal example.
- The base model and fine-tuning data are English-focused; multilingual, code-switched, or dialect-heavy inputs may be unreliable.
- The model is costly to train and comparatively heavy to deploy.
- Genre and domain shifts can reduce accuracy.
- Authorship predictions can enable unwanted deanonymization. They should not be treated as conclusive identity evidence.

## Reproduction

See the [training notebook](../notebooks/roberta-asymmetric-loss/training.ipynb), [demo notebook](../notebooks/roberta-asymmetric-loss/demo.ipynb), and [reproduction guide](../docs/reproduction.md).
