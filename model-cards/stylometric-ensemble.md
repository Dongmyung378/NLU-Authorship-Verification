---
language: en
tags:
  - text-classification
  - authorship-verification
  - stylometry
  - xgboost
  - lightgbm
repo: https://github.com/Dongmyung378/NLU-Authorship-Verification
---

# Stylometric XGBoost-LightGBM Ensemble

[한국어](./stylometric-ensemble.ko.md) | [Project overview](../README.md)

This card describes the original stylometric development artifact. Its scores are historical development results and are not a controlled comparator for models trained on the new document-disjoint split.

## Model summary

This supervised binary classifier was trained to label pairs in the supplied English text dataset as same-author or different-author. It combines word-, character-, and function-word TF-IDF signals with handcrafted stylometric features, then blends XGBoost and LightGBM outputs using tuned ensemble weights and a tuned decision threshold. Its reported performance is limited to the supplied development split.

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

The model used the 27,643 labeled English text pairs in the supplied [`train.csv`](../data/train.csv); no additional labeled pairs were added for task-specific training. The labeled train and development CSVs contain passages and pair labels, but no author IDs or domain labels, so an author-separated or domain-separated evaluation cannot be verified from these files.

The original training notebook fits all three TF-IDF vectorizers on concatenated cleaned train and development passages (`all_txt`). Development text therefore influences vocabulary and IDF even though its labels are not used for vectorizer fitting. This is a historical feature-fitting limitation alongside development-based model selection. The new document-disjoint study fits every vectorizer on train only; its separately frozen results must not be compared with the historical development score as a controlled before-and-after gain.

## Hyperparameters

### XGBoost

| Parameter | Value |
|---|---:|
| Estimator ceiling | 7,000 |
| Saved best iteration | 6,013, zero-based; 6,014 boosting rounds |
| Retained trees in the saved artifact | 6,214 |
| Learning rate | 0.01 |
| Max depth | 7 |
| Subsample | 0.8 |
| Column sample by tree | 0.8 |
| Minimum child weight | 2 |
| Max bins | 128 |
| Early stopping patience | 200 rounds |

The saved `xgb_model.json` records `best_iteration=6013` and contains 6,214 trees. Early stopping can retain trees after the best iteration, so the selected best round and retained tree count are distinct. The earlier card's 6,213 value was not the saved best iteration.

### LightGBM

| Parameter | Value |
|---|---:|
| Estimator ceiling | 2,000 |
| Saved best iteration | 1,671 |
| Trees in the saved artifact | 1,671 |
| Learning rate | 0.01 |
| Leaves | 127 |
| Subsample | 0.8 |
| Subsample frequency | 1 |
| Column sample by tree | 0.8 |
| Max bins | 127 |
| Early stopping patience | 100 rounds |

A static inspection of the saved LightGBM pickle, without executing it, found `_best_iteration=1671`; its embedded model text contains 1,671 trees. The earlier card's 1,700 stopping point is not established by that saved artifact. The selected model size and a later training stopping point should not be conflated.

### Ensemble

| Setting | Value |
|---|---:|
| XGBoost weight | 0.26 |
| LightGBM weight | 0.74 |
| Decision threshold | 0.43 |

## Evaluation

Evaluation used 5,993 development pairs: 2,937 different-author pairs and 3,056 same-author pairs. A comparison of the supplied CSVs found 630 development pairs (10.5%) that exactly repeat a text pair in the training split, with matching labels. The development split was also used for early stopping, ensemble weights, and the decision threshold. These figures are development results and do not establish performance on independent, unseen authors or domains.

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
- [Download trained artifacts](https://drive.google.com/drive/folders/1mF0bMpzfs3XODpQA2Hd7H_JvXLk5AGgp?usp=sharing). The downloaded model files are kept outside GitHub; the repository ignores `models/`.

## Run the downloaded model

From the repository root, download all files from the link above and put them directly in `models/Models_ML/` (the original download folder name) or `models/stylometric-ensemble/`. The directory must contain `ensemble_config.json`, `xgb_model.json`, `lgb_model.pkl`, `tfidf_word.pkl`, `tfidf_char.pkl`, and `tfidf_fw.pkl`. Avoid an extra nested folder between the chosen directory and these files.

Install the [terminal demo dependencies](../requirements-inference.txt), then enter two English passages when prompted:

```bash
python -m pip install -r requirements-inference.txt
python verify_author.py --model stylometric
```

To pass text on the command line, use `python verify_author.py --model stylometric --text-1 "First English passage" --text-2 "Second English passage"`. This pipeline runs on CPU. The output includes `Same-author probability`, the loaded model's decision threshold, and the resulting label. Despite the displayed name, this blended classifier output has not been calibrated as the probability of a shared author. Results for arbitrary new authors or domains have not been established. If the files live elsewhere, add `--model-dir PATH` with the directory that directly contains them. See the [reproduction guide](../docs/reproduction.md) for environment setup and notebook use.

## Limitations and risks

- The feature set is designed for English; multilingual or code-switched inputs are not supported reliably.
- Style features become less informative on passages shorter than five words.
- Short passages may be unreliable: the median training passage has about 123 whitespace-separated words, much longer than a short terminal example.
- TF-IDF ignores terms outside the fitted vocabulary.
- Genre shifts can change stylistic patterns and reduce accuracy.
- Authorship predictions can enable unwanted deanonymization. They should not be treated as conclusive identity evidence.

## Reproduction

See the [training notebook](../notebooks/stylometric-ensemble/training.ipynb), [demo notebook](../notebooks/stylometric-ensemble/demo.ipynb), and [reproduction guide](../docs/reproduction.md).
