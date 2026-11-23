# Architecture

[한국어](./architecture.ko.md) | [README](../README.md)

The project separates traditional stylometry (`Track 01`) and neural representation learning (`Tracks 02 & 03`) so both paradigms can be evaluated without cross-paradigm feature or score leakage. Every track enforces exact order symmetry (`P(A, B) = P(B, A)`) and inductive single-pair inference.

---

## 1. Track 01: 6-Layer Stylometric ML (`01_stylometric_ml`)

### Feature Extraction (`6,001` dimensions = `501` dense + `5,500` sparse)

Each passage is decomposed into three complementary linguistic views:
- **Surface view**: Character `2-5`-grams (preserving punctuation, capitalization, and morphological endings).
- **POS-Noise view**: Content words are replaced by `<POS>_<suffix>` tags while keeping 250 function words, numbers (`<NUM>`), and punctuation (`P_,`) intact to break topic shortcuts.
- **Distorted view**: Content words are masked by length and capitalization pattern (`w5`, `C7`) to capture rhythmic structure independent of topic vocabulary.

| Layer | Features | Construction | Combined Importance |
| :--- | :---: | :--- | :---: |
| `G1` Stage 1 Margin & Gap | 18 | 5-bucket dual author-disjoint OOF linear SVM/LR on `60k/40k/30k` views + `POS-Noise - Surface` gap | 38.74% |
| `G2` Impostor Cohort & AS-Norm | 14 | 256 frozen training impostors: `AS-Norm`, Top-10 margin, mutual rank, `64-Bag Koppel` win rate | 22.54% |
| `G6` Multi-View Sparse Habits | 5,515 | Symmetric `\|v1 - v2\|` and `v1 * v2` on `3~5` char n-grams and `1~3` POS-Noise n-grams + kernel similarities | 21.28% |
| `G4` Gram2Vec & Biometrics | 190 | 62 syntactic, POS-bigram, punctuation, MATTR 50, Yule's K, burstiness, and readability features (abs/rel diff + pair mean) | 12.15% |
| `G3` Compression Distance | 6 | Symmetric `zlib` Normalized Compression Distance (`NCD`) and `CBC` on raw and POS-Noise text | 4.34% |
| `G5` Function Words & Delta | 258 | 250 function-word Z-score differences + Burrows' / Argamon / Cosine Delta + JSD | 0.95% |

### 10-Expert Multi-View Ensemble + OOD Register Gate

1. **Full-view experts (3)**: `LightGBM GBDT`, `LightGBM GOSS`, `CatBoost Dense`.
2. **Pruned + interaction experts (5)**: Top-`1,500` gain-selected columns + 12 symmetric cross-view consensus features (`LightGBM GBDT/GOSS`, `CatBoost Hybrid 662`, `CatBoost Depth 4`, `L2 Logistic`).
3. **Domain-debiased experts (2)**:
   - `Genre-Blind Pure Delta CatBoost`: `642` pure difference columns with zero surface vocabulary and zero pair-mean genre cues.
   - `Cross-Topic 2.5x Weighted LightGBM`: Upweights cross-topic pairs (`2.5x`) and applies inverse-sqrt domain balancing.
4. **Intrinsic OOD Register-Gated Meta-Calibrator**: Combines the 10 expert logits with intrinsic single-pair OOD signals (impostor top-10 affinity, surface inflation ratio, formality indices, and `Genre-Blind - Surface` disagreement) via 5-bucket dual author-disjoint OOF `0.45 LR + 0.55 CatBoost`.

---

## 2. Tracks 02 & 03: Symmetric Siamese DeBERTa-v3 (`12L` / `24L`)

### Backbone & Pooling

- **Token budget**: `512` tokens per document (`1,024` combined), encoded symmetrically by a shared DeBERTa-v3 encoder (`FP32` master weights under `BF16` autocast).
- **Weighted Top-50% Layer Pooling**: Softmax-weighted aggregation of upper layers (`L7~L12` on Base, `L13~L24` on Large).
- **Attentive Statistics Pooling (`ASP`)**: Attention-weighted token mean concatenated with attention-weighted token standard deviation (`2 * H`), capturing stylistic rhythm variance.
- **Symmetric Interaction Head**: Maps `(u, v)` via `[0.5*(u+v), |u-v|, u_unit*v_unit, (u_unit-v_unit)^2, cos(u_unit, v_unit)]` through a LayerNorm-GELU MLP and 5-branch Multi-Sample Dropout (`p = 0.10..0.30`).

### 4-Depth Multi-Std + Dual-Rank `All-but-the-Top` Residual OOD Gate (Track 03)

To remove shared-topic and unseen-genre anisotropy without domain labels or ML features:
1. Extract 4 linguistic depth bands from the 24-layer backbone:
   - **Band 1 (`L2~L6`)**: Lexical, punctuation, and orthographic habits (mean + token std `s1`).
   - **Band 2 (`L8~L12`)**: Syntactic structure and function-word rhythm (mean + token std `s2`).
   - **Band 3 (`L14~L18`)**: Local discourse flow (mean + token std `s3`).
   - **Band 4 (`L22~L24`)**: High-level semantic/topic representation (mean).
2. Fit **Dual-Rank `All-but-the-Top`** (`r=3` mild topic removal and `r=8` aggressive topic removal) strictly via SVD on `256` frozen training impostors, then compute symmetric `AS-Norm` and top-10 margins across all bands.
3. Combine `71` scalar geometric/depth features with `36`-dim centroid PCA (`0.5*(e1+e2)`), `10` raw-logit x centroid interactions, and `16`-dim **symmetric agreement PCA** (`e1 * e2` on `r=8` embeddings).
4. Fit a 5-bucket dual author-disjoint OOF ensemble (`0.35 LR(C=0.06) + 0.65 5-Seed ResidualOODGate`), exported as pure PyTorch tensors (`model_03_deberta_v3_large.pt`) with zero `scikit-learn` pickle dependencies.
