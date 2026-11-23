# Model Card: Track 03 / DeBERTa-v3-large (`03_deberta_v3_large`)

[한국어](./deberta-v3-large.ko.md) | [README](../README.md)

- **Artifact**: `model_03_deberta_v3_large.pt` (`860.27 MB`, `436.64M` params, pure PyTorch tensors including FP16 backbone weights, 256 impostor 4 depth representations, Dual Rank ABTT matrices, and 5-fold 5-seed `ResidualOODGate` weights / zero `sklearn` pickles)
- **Download**: [Google Drive](https://drive.google.com/file/d/1r19ezaf0Q_nn3qVS43m4nDhAs5jfGkve/view) · [Colab Notebook](https://colab.research.google.com/drive/1ROS0vdfbpZupVoM38Dz41eDB1tCAGjWL)
- **Backbone & Recipe**: `microsoft/deberta-v3-large` (24L), `LLRD (gamma=0.94868) + Prior WD (0.01) + Pure Label Smoothed BCE`, interpolated checkpoint soup (`0.65 * step_4141 + 0.35 * step_4970`), 4 depth Multi Std (`L2~L6`, `L8~L12`, `L14~L18` mean+std + `L22~L24`), Dual-Rank `All but the Top (r=3, 8)`, symmetric agreement PCA (`e1 * e2`, 16d).
- **Input**: Pair of English passages (`<= 512` tokens each, `1,024` combined budget), strictly order-invariant (`|P(A,B) - P(B,A)| = 0.00`).
- **Decision Policy**: Threshold `tau = 0.5913`, PAN abstention half-width `delta = 0.1360`, inductive `CSLS hw = 1.15`.
- **Locked Test Performance**:
  - Closed 1:30 (`240` queries): `Top 1 78.33%`, `Top 3 91.25%`, `Macro F1 0.6547` (`imdb_reviews 91.67%/99.17%`, `news 73.33%/93.33%`)
  - Open + Cross Domain (`9,440` pairs): `ROC AUC 0.9139`, `PAN c@1 0.8556`, `PAN F_0.5u 0.8461`, `PAN Brier 0.8779`, `PAN Overall 0.8643`, `Forced Macro F1 0.8271`
  - Zero Shot `arxiv_research`: `ROC AUC 0.9389`, `FPR 8.6%`
  - Ultra Short `<60w/doc`: `ROC AUC 0.9285` (`+9.98%p` vs ML)
- **Limitations**: Higher compute footprint than Track 01 (GPU recommended: `32.5 ms/pair` on L4, `266 ms/pair` on RTX 3060); `cross_genre` journalism (`0.7344`) remains more challenging than same-genre or zero shot academic verification. Passages beyond 512 tokens are truncated.
