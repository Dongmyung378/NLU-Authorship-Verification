# Design Decisions, Improvement Progression, and Troubleshooting

[한국어](./methodology-and-decisions.ko.md) | [README](../README.md)

This document records **why** each model, feature group, training hyperparameter, and post hoc calibration method was chosen, **what failed and was rejected**, and **how every diagnostic round moved validation performance** prior to evaluating the locked test set once.

---

## 1. Benchmark & Evaluation Decisions

| Decision | Why It Was Made | Alternative Rejected |
| :--- | :--- | :--- |
| **Dual-author-disjoint split (`benchmark_1024`)** | Earlier splits leaked shared passages/authors across train and validation, inflating F1 through document memorization. Zero author overlap isolates genuine authorship verification. | Random pair split or document-only split without author IDs. |
| **5 bucket dual author disjoint OOF** | Standard `GroupKFold(groups=author_1)` still leaks `author_2` across folds. Requiring `(b(a1) != k) & (b(a2) != k)` guarantees zero author leakage in Stage 1 and meta-calibrators. | Single-side `GroupKFold` or `StratifiedKFold`. |
| **Strict No Hybrid separation** | Keeping Pure-ML and Pure-DL completely separate quantifies the exact scaling gain of neural encoders vs explicit stylometry and reveals where ML still outperforms DL (`cross_genre`). | Blending ML features into DeBERTa logits. |
| **Inductive single pair scoring** | Batch-level transductive normalization cannot reproduce benchmark scores on a single live pair. Freezing 256 training impostors and intrinsic anchor hubness priors guarantees live demo parity (`< 1e-4` ML, `< 4e-3` DL FP16). | Test-batch Z-score normalization or transductive CSLS. |

---

## 2. Track 01 (`Stylometric ML`): 3 Round Iterative Progression

| Stage | Val Closed Top 1 | Val Open ROC AUC | Val PAN Overall | Key Diagnosis & Engineering Change |
| :--- | :---: | :---: | :---: | :--- |
| Legacy Surface Single LightGBM | 48.67% | 0.7632 | 0.7336 | Surface n grams overfit to shared topics. |
| Round 1: 6 Layer Suite Heterogeneous Ensemble | 60.00% | 0.8009 | 0.7520 | Added 3-view `POS Noise`/`Distorted`, 256 impostors, Koppel 64 Bag, Gram2Vec, NCD. |
| Round 2: Temp-Calibrated 8 Expert + Pruning | 62.00% | 0.8005 | 0.7594 | Diagnosed uncalibrated LGBM logit scale (`±6`) collapsing weights to `0.005`; added in-fold temperature alignment + pruned `1,702` zero-gain features (`6,001 → 1,500`) + 12 consensus interactions. |
| **Round 3: 10 Expert + OOD Register Gate (`d=0.068`)** | **67.33%** | **0.8378** | **0.7929** | Diagnosed `89.6%` (`Guardian`) and `75.4%` (`ArXiv`) FPR from formal-register illusion; added Genre Blind Pure Delta CatBoost (`642` cols), Cross Topic `2.5x` LGBM, and intrinsic OOD Register Gate (zero domain labels). |
| **Locked Test (frozen thresholds)** | **64.58%** | **0.8474** | **0.8008** | Guardian & ArXiv Test FPR both dropped to `25.7%`; zero validation overfitting (`+0.96%p` AUC). |

**Systems fixes implemented in Track 01:**
- **OpenBLAS / OpenMP contention (`15 min → <0.5 s`)**: Wrapping `scikit-learn` L-BFGS fits inside `threadpool_limits(limits=1)` eliminated Linux thread deadlock after LightGBM/CatBoost training.
- **Cross-version portable serialization (`21.63 MB`)**: Replaced NumPy/sklearn pickles with pure Python builtins (`bytes`, `str`, `dict`) + LightGBM text + CatBoost `.cbm`, and compressed the `245.8 MB` Koppel random projection matrix into its `150`-byte `PCG64` RNG state (exact `0.000000` reconstruction).

---

## 3. Track 02 (`DeBERTa-v3-base`): Troubleshooting, 4 Step Ablation & 4-Round OOD Calibration

### Root Cause: `transformers 5.x` Silent FP16 Underflow
Initial `DeBERTa-v3-base` training stayed flat at `loss = 0.6965` (`Val AUC = 0.5152`). Disabling `autocast` raised `RuntimeError: expected Half and Float`, revealing that `transformers 5.x` loads DeBERTa-v3's HuggingFace Hub `float16` weights directly in FP16, causing `AdamW` updates (`1.6e-5`) to round to zero below FP16 machine epsilon (`~1e-3`). Calling `AutoModel.from_pretrained(...).float()` with an explicit FP32 assertion immediately restored normal convergence.

### 4 Step Controlled Ablation (`40%` data = `33,944` pairs x `1.5` epochs)

| Step | Val Closed Top 1 | Val Open ROC AUC | Val PAN Overall | Verdict & Root Cause |
| :--- | :---: | :---: | :---: | :--- |
| `Step 0`: Vanilla Siamese Baseline | 67.33% | 0.8005 | 0.7690 | Baseline. |
| `Step 1`: + Top 6 Layers + ASP + MSD + 15% EntityMask | 66.67% | **0.8083** | **0.7697** | **KEEP**: improves Open AUC (`+0.78%p`) and Closed Macro F1. |
| `Step 2`: + PriorWD + LLRD + CoSENT (`shuffle=False`) | 60.67% | 0.7927 | 0.7547 | **DROP**: domain sorted `shuffle=False` batches caused sequential forgetting across domains. |
| `Step 3`: + Greedy Soup + EMA + AS Norm + CSLS | 62.00% | **0.8098** | **0.7785** | **KEEP (decoupled S1)**: lifted Open PAN Overall `+0.95%p`; decoupled raw cosine AS Norm from Closed 1:30 matching so S1 selects best validation logit mode. |

### 4-Round Pure-Neural Post Hoc Progression (Zero Retraining)
After full 2.5-epoch training (`shuffle=True`), `Base` reached `news AUC 0.9447` but collapsed on zero shot genres (`Guardian FPR 97.6%`, `ArXiv FPR 94.8%`, overall Val AUC `0.8155`).
1. **Round 2 (`02c`: Neural OOD Manifold Gate)**: Impostor top-10 affinity + L2 norms + `PCA(24)` + PyTorch `ResidualOODGate` → Val AUC **`0.8155 → 0.8912` (`+7.57%p`)**, `Guardian FPR -60%p`, `ArXiv FPR -67%p`.
2. **Round 3 (`02e`: Mid-layer `L4-6` syntax + mean-centering)**: Added syntax-vs-semantic layer gap → Val AUC **`0.8956`**, Closed Top 1 `76.00%`.
3. **Round 4 (`02f`: 4 Depth + `All but the Top r=3`)**: Decomposed `L1-3`, `L4-6 mean+std`, `L7-9`, `L11-12` and removed top-3 SVD topic axes → Val AUC **`0.9009`**, Closed Top 1 **`77.33%`**, Locked Test **`0.9040` / `76.25%`**.

---

## 4. Track 03 (`DeBERTa-v3-large`): HP Probe, Deferred Snapshot Soup & Dual Rank ABTT

### Why `LLRD + Prior WD + Pure BCE` Won on 24 Layers
Re-testing `LLRD + Prior WD` **with `shuffle=True`** on a 10% stratified probe (`8,486` pairs) proved that the failure in Base Step 2 was caused by `shuffle=False`, not LLRD/Prior WD:
- `Probe_B (lr=1.6e-5 / 1.0e-4, LLRD gamma=0.94868, Prior WD 0.01, Pure BCE)`: **Val Composite `0.71345`** (`Closed Top 1 62.67%`, `Open AUC 0.78423`) / selected.
- `Probe_A (1.2e-5, LLRD + Prior WD, Pure BCE)`: `0.69180`.
- `Probe_C (1.2e-5, LLRD + Prior WD, CoSENT 0.08)`: `0.69024` / confirmed CoSENT adds no gain over label smoothed BCE.
- `Probe_D (1.0e-5, Uniform LR)`: `0.67642` / lagged LLRD+Prior WD by `-3.70%p`.

### L4 24GB VRAM Engineering & Final Card (`0.65/0.35` Soup + Dual Rank ABTT + AgreePCA)
- **Deferred Snapshot Evaluation**: Evaluating snapshots mid-training while optimizer states and Prior WD anchors stayed on GPU overflowed `L4 22.03 GB`. Saving snapshots (`50%, 62.5%, 75%, 87.5%, 100%`) directly to CPU disk and freeing optimizer/anchors before evaluation kept peak VRAM safe (`~19 GB`) with zero recomputation loss.
- **Why Interpolated Soup (`0.65 * step_4141 + 0.35 * step_4970`) + Dual Rank ABTT (`r=3, 8`) + Multi Std (`s1,s2,s3`) + Symmetric Agreement PCA (`e1*e2`)**:
  - `R2` baseline (`step_4141 + 4 Depth r=3`): Val Open AUC `0.91009`, PAN Overall `0.86368`, `ArXiv 0.8998`, `Guardian 0.7642`.
  - Adding Dual Rank ABTT (`r=3` & `r=8`, exposing the topic removal sensitivity gap `an_r8 - an_r3`), token std across all three lower bands (`s1/s2/s3`), and symmetric agreement PCA (`e1 * e2`, 16d) lifted Val Open AUC to `0.91237` and `ArXiv` to `0.9080`.
  - Interpolating the top two checkpoints (`0.65 * step_4141 + 0.35 * step_4970`) with extended inductive CSLS (`hw=1.15`) achieved the best overall balance: **Val Open AUC `0.91388`**, **Locked Test Open AUC `0.91393`** (gap `0.00005`), **Closed Top 1 `78.33%`**, and **Zero Shot `ArXiv` AUC `0.9389` (`FPR 8.6%`)**.
