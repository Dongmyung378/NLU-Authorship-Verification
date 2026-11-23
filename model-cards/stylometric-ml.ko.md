# 모델 카드: Track 01 / Stylometric ML (`01_stylometric_ml`)

[English](./stylometric-ml.md) | [README](../README.ko.md)

- **아티팩트**: `model_01_stylometric_ml.joblib` (`21.63 MB`, 순수 Python 내장 타입 + LightGBM 문자열 + CatBoost `.cbm`, NumPy/sklearn 버전 의존성 없음)
- **다운로드**: [Google Drive](https://drive.google.com/file/d/1q9T8zLYmnhLqPWT9He8wiqIjA0kvCR1M/view) · [Colab 학습 노트북](https://colab.research.google.com/drive/1jAuZKPe7mcuL9jB9SAIQmGr-jQ-CdjtJ)
- **패러다임**: 순수 문체학 특징 공학 + 결정 트리/선형 메타 앙상블 (신경망 임베딩 미사용)
- **입력**: 영어 글 두 편, 순서 불변 대칭 추론 (`|P(A,B) - P(B,A)| = 0.00`)
- **판정 기준**: 임계값 `tau = 0.5762`, PAN 보류 구간 `delta = 0.0680`, 귀납적 `CSLS hw = 0.35`
- **Final Locked Test 성적**:
  - 닫힌 집합 1:30 (`240` 쿼리): `Top-1 64.58%`, `Top-3 84.58%`, `Macro F1 0.5359`
  - 오픈셋·교차 도메인 (`9,440`쌍): `ROC AUC 0.8474`, `PAN c@1 0.7819`, `PAN Brier 0.8387`, `PAN Overall 0.8008`, `Forced Macro F1 0.7628`
  - 강점: GPU 없이 CPU `32.1 ms/pair` 추론, 문체 지표별 해석 가능성, `cross_genre` (`0.8019`) 및 `guardian_journalism` (`0.8024`) 1위.
- **한계**: 60단어 미만 초단문에서는 빈도 통계가 희소해져 `DeBERTa-v3-large` 대비 낮습니다(`0.8287` vs `0.9285`). 법의학적 신원 입증 용도로 사용할 수 없습니다.
