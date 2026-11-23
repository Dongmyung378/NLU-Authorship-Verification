# Benchmark Results & Statistical Verification

[한국어](./results.ko.md) | [README](../README.md)

All metrics below are measured on the locked test split of `benchmark_1024` (`240` closed-set `1:30` queries / `18,000` pairs and `9,440` open-set / cross topic / zero shot pairs, zero shared authors with Train or Validation).

---

## 1. Three-Track Locked Test Summary

| Track | Closed Top 1 | Closed Top 3 | Closed Macro F1 | Open ROC AUC | PAN c@1 | PAN F_0.5u | PAN Brier | PAN Overall | Forced Macro F1 | Answered Acc | Abstain % |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `01. Stylometric ML` | 64.58% | 84.58% | 0.5359 | 0.84740 | 0.78192 | 0.78225 | 0.83867 | 0.80082 | 0.76280 | 79.80% | 14.17% |
| `02. DeBERTa-v3-base` | 76.25% | **91.25%** | 0.6363 | 0.90399 | 0.84428 | 0.84144 | 0.87237 | 0.85669 | 0.81958 | 86.12% | 14.04% |
| **`03. DeBERTa-v3-large`** | **78.33%** | **91.25%** | **0.6547** | **0.91393** | **0.85556** | **0.84611** | **0.87789** | **0.86430** | **0.82711** | **87.57%** | 15.15% |

## 2. Exact McNemar & Paired Bootstrap 95% CI

| Task | Comparison | A Wins (`n10`) | B Wins (`n01`) | Exact McNemar `p` | Sig (`p<0.05`) | Delta Acc (95% CI) | Delta AUC (95% CI) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Scenario 1 (240q) | `Base` vs `ML` | 46 | 18 | `6.17e-04` | Yes | `+11.67% [+5.42%, +17.92%]` | N/A |
| Scenario 1 (240q) | `Large` vs `ML` | 53 | 20 | `1.42e-04` | Yes | `+13.75% [+7.08%, +20.83%]` | N/A |
| Scenario 1 (240q) | `Large` vs `Base` | 20 | 15 | `0.4996` | No | `+2.08% [-2.50%, +7.08%]` | N/A |
| Scenario 2 (9,440p) | `Base` vs `ML` | 1,399 | 866 | `2.89e-29` | Yes | `+5.65% [+4.66%, +6.62%]` | `+5.66% [+4.90%, +6.36%]` |
| Scenario 2 (9,440p) | `Large` vs `ML` | 1,450 | 846 | `1.01e-36` | Yes | `+6.40% [+5.40%, +7.38%]` | `+6.65% [+5.90%, +7.39%]` |
| **Scenario 2 (9,440p)** | **`Large` vs `Base`** | **601** | **530** | **`0.03735`** | **Yes** | **`+0.75% [+0.08%, +1.45%]`** | **`+0.99% [+0.63%, +1.37%]`** |

## 3. Breakdown by Domain, Stratum, and Passage Length (ROC AUC)

| Category / Slice | Pairs | `01. Stylometric ML` | `02. DeBERTa-v3-base` | `03. DeBERTa-v3-large` |
| :--- | :---: | :---: | :---: | :---: |
| Stratum: `open_unseen_author_same_topic` | 5,840 | 0.8473 | 0.9236 | **0.9339** |
| Stratum: `cross_topic` | 2,158 | 0.8340 | 0.8787 | **0.8811** |
| Stratum: `zero_shot_unseen_genre` (`arxiv`) | 880 | 0.9336 | 0.9093 | **0.9389** |
| Stratum: `cross_genre` (`guardian` etc.) | 562 | **0.8019** | 0.7385 | 0.7344 |
| Length: `Ultra Short (<60w/doc)` | 4,058 | 0.8287 | 0.9157 | **0.9285** |
| Length: `Short (60~180w/doc)` | 2,226 | 0.8848 | 0.9186 | **0.9357** |
| Length: `Medium (180~350w/doc)` | 3,156 | 0.8421 | **0.8752** | 0.8739 |
| Domain: `news` | 2,076 | 0.8362 | 0.9538 | **0.9603** |
| Domain: `blogs` | 1,714 | 0.8298 | 0.8964 | **0.9137** |
| Domain: `guardian_journalism` | 1,440 | **0.8024** | 0.7692 | 0.7513 |
| Domain: `imdb_reviews` | 1,440 | 0.8743 | 0.9397 | **0.9468** |
| Domain: `social_media` | 1,304 | 0.8994 | 0.9204 | **0.9322** |
| Domain: `arxiv_research` | 880 | 0.9336 | 0.9093 | **0.9389** |
| Domain: `reviews_ecommerce` | 508 | 0.7332 | 0.8037 | **0.8313** |

## 4. Standalone Artifact Parity Verification

- **`model_01_stylometric_ml.joblib` (`21.63 MB`, pure Python builtins)**: Windows CPU `32.1 ms/pair`, order symmetry `0.00e+00`, reload diff against Locked Test CSV `1.41e-06`.
- **`model_03_deberta_v3_large.pt` (`860.27 MB`, pure PyTorch tensors)**: Local RTX 3060 Laptop GPU `265.97 ms/pair` (Colab L4 `32.45 ms/pair`), `strict=True` load (`missing=0, unexpected=0`), `24/24 (100%)` binary agreement on diverse Locked Test pairs, mean `|p_pt - p_colab| = 0.001147` (max `0.003964`), order symmetry `0.00e+00`.
