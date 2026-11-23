# Model Card: Track 01 / Stylometric ML (`01_stylometric_ml`)

[한국어](./stylometric-ml.ko.md) | [README](../README.md)

- **Artifact**: `model_01_stylometric_ml.joblib` (`21.63 MB`, pure Python builtins + LightGBM strings + CatBoost `.cbm` blobs, zero NumPy/sklearn pickle version coupling)
- **Download**: [Google Drive](https://drive.google.com/file/d/1q9T8zLYmnhLqPWT9He8wiqIjA0kvCR1M/view) · [Colab Notebook](https://colab.research.google.com/drive/1jAuZKPe7mcuL9jB9SAIQmGr-jQ-CdjtJ)
- **Paradigm**: Feature-engineered stylometric machine learning (zero neural embeddings)
- **Input**: Pair of English passages (`<= 350 words` target range), strictly order-invariant (`|P(A,B) - P(B,A)| = 0.00`)
- **Decision Policy**: Threshold `tau = 0.5762`, PAN abstention half-width `delta = 0.0680`, inductive `CSLS hw = 0.35`
- **Locked Test Performance**:
  - Closed 1:30 (`240` queries): `Top 1 64.58%`, `Top 3 84.58%`, `Macro F1 0.5359` (`imdb_reviews 78.33%`, `blogs 60.00%`)
  - Open + Cross Domain (`9,440` pairs): `ROC AUC 0.8474`, `PAN c@1 0.7819`, `PAN Brier 0.8387`, `PAN Overall 0.8008`, `Forced Macro F1 0.7628`
  - Key strengths: CPU inference `32.1 ms/pair`, zero GPU required, interpretable per-feature attribution, highest ROC AUC on `cross_genre` (`0.8019`) and `guardian_journalism` (`0.8024`).
- **Limitations**: Lower accuracy than 24 layer DeBERTa-v3-large on ultra-short passages (`<60 words/doc`, `0.8287` vs `0.9285`) where lexical/syntactic frequencies are sparse. Not intended as forensic identity proof.
