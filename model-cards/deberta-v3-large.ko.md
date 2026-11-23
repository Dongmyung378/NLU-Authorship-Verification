# 모델 카드: Track 03 — DeBERTa-v3-large (`03_deberta_v3_large`)

[English](./deberta-v3-large.md) | [README](../README.ko.md)

- **아티팩트**: `model_03_deberta_v3_large.pt` (`860.27 MB`, `436.64M` 파라미터, FP16 백본 + 256 사칭자 4계층 표현 + Dual-Rank ABTT 직교 행렬 + 5-Fold 5-Seed `ResidualOODGate` 순수 텐서, `sklearn` 피클 미사용)
- **다운로드**: [Google Drive](https://drive.google.com/file/d/1r19ezaf0Q_nn3qVS43m4nDhAs5jfGkve/view) · [Colab 학습 노트북](https://colab.research.google.com/drive/1ROS0vdfbpZupVoM38Dz41eDB1tCAGjWL)
- **백본 및 레시피**: `microsoft/deberta-v3-large` (24층), `LLRD (gamma=0.94868) + Prior-WD (0.01) + Pure Label-Smoothed BCE`, 체크포인트 보간 수프(`0.65 * step_4141 + 0.35 * step_4970`), 4계층 Multi-Std + Dual-Rank `All-but-the-Top (r=3, 8)` + 대칭 합의축 PCA(`e1 * e2`).
- **입력**: 영어 글 두 편 (문서당 최대 `512`토큰, 합산 `1,024`토큰), 순서 불변 대칭 추론 (`|P(A,B) - P(B,A)| = 0.00`).
- **판정 기준**: 임계값 `tau = 0.5913`, PAN 보류 구간 `delta = 0.1360`, 귀납적 `CSLS hw = 1.15`.
- **Final Locked Test 성적**:
  - 닫힌 집합 1:30 (`240` 쿼리): `Top-1 78.33%`, `Top-3 91.25%`, `Macro-F1 0.6547`
  - 오픈셋·교차 도메인 (`9,440`쌍): `ROC-AUC 0.9139`, `PAN c@1 0.8556`, `PAN F_0.5u 0.8461`, `PAN Brier 0.8779`, `PAN Overall 0.8643`, `Forced Macro-F1 0.8271`
  - 제로샷 `arxiv_research`: `ROC-AUC 0.9389`, `FPR 8.6%`
  - 초단문 `<60w/doc`: `ROC-AUC 0.9285` (`ML` 대비 `+9.98%p`)
- **한계**: CPU 추론보다 GPU 환경(L4 `32.5 ms/pair`, RTX 3060 `266 ms/pair`)이 적합하며, `cross_genre` 저널리즘(`0.7344`)은 전통 문체학 ML 대비 낮습니다. 문서당 512토큰 초과 부분은 잘립니다.
