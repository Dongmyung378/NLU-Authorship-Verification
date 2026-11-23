# 시스템 구조

[English](./architecture.md) | [README](../README.ko.md)

본 프로젝트는 전통 문체학(`Track 01`)과 순수 딥러닝(`Tracks 02 & 03`)을 엄격히 분리하여 특징이나 예측값을 혼합하지 않고 비교합니다. 모든 트랙은 입력 순서 불변성(`P(A, B) = P(B, A)`)과 단일 글 쌍 귀납적 추론을 보장합니다.

---

## 1. Track 01: 6계층 문체학 ML (`01_stylometric_ml`)

### 특징 추출 (`6,001`차원 = `501` Dense + `5,500` Sparse)

각 문서를 서로 다른 세 가지 뷰로 분해합니다.
- **Surface 뷰**: 대소문자와 구두점을 보존한 문자 `2-5`-gram.
- **POS Noise 뷰**: 250개 기능어, 숫자(`<NUM>`), 구두점(`P_,`)만 남기고 내용어를 `<품사>_<접미사>`로 치환해 주제 어휘 지름길을 차단.
- **Distorted 뷰**: 내용어를 대소문자와 글자 수 패턴(`w5`, `C7`)으로 마스킹해 문장 리듬을 포착.

| 계층 | 피처 수 | 구성 방식 | 통합 기여도 |
| :--- | :---: | :--- | :---: |
| `G1` Stage 1 마진·괴리도 | 18 | `60k/40k/30k` 3중 뷰 5 Bucket 양방향 저자 배제 OOF 선형 SVM/LR + `POS Noise - Surface` 괴리도 | 38.74% |
| `G2` 사칭자 코호트 & AS Norm | 14 | 학습 고정 사칭자 256편 기준 `AS Norm`, Top 10 마진, 상호 순위, `64 Bag Koppel` 승률 | 22.54% |
| `G6` 다중 뷰 희소 문체 습관 | 5,515 | 철자 `3~5` gram 및 `POS Noise 1-3`-gram 대칭 `\|v1 - v2\|`, `v1 * v2` 및 커널 유사도 | 21.28% |
| `G4` Gram2Vec & 생체 리듬 | 190 | 구문·품사 전이·구두점·어휘 다양성(`MATTR 50`, `Yule's K`)·가독성 62종(절대차·상대차·쌍 평균) | 12.15% |
| `G3` 압축 거리 | 6 | 원문 및 `POS Noise` 시퀀스에 대한 대칭 `zlib` 압축 거리(`NCD`, `CBC`) | 4.34% |
| `G5` 기능어 & Burrows' Delta | 258 | 영어 기능어 250종 Z-편차 + `Burrows' / Argamon / Cosine Delta` + `JSD` | 0.95% |

### 10전문가 다중 뷰 앙상블 + 내재적 OOD 레지스터 게이트

1. **전체 뷰 전문가 (3종)**: `LightGBM GBDT`, `LightGBM GOSS`, `CatBoost Dense`.
2. **가지치기·교차 합의 전문가 (5종)**: Gain 상위 `1,500`열 + 12종 대칭 교차 합의 피처 (`LightGBM GBDT/GOSS`, `CatBoost Hybrid 662`, `CatBoost Depth 4`, `L2 Logistic`).
3. **도메인 편향 억제 전문가 (2종)**:
   - `Genre Blind Pure Delta CatBoost`: 표면 단어와 쌍 평균 장르 지표를 제외하고 순수 문체 편차 `642`열로만 학습.
   - `Cross Topic 2.5x Weighted LightGBM`: 교차 주제 쌍 가중치 `2.5x` 및 도메인 역제곱근 균형 적용.
4. **내재적 OOD 레지스터 게이트**: 10개 전문가 로짓에 단일 쌍 내부의 사칭자 친화도, 표면 과장 배율, 문체 격식도, `Genre Blind - Surface` 불일치도를 결합해 5 Bucket 양방향 저자 배제 OOF `0.45 LR + 0.55 CatBoost`로 보정.

---

## 2. Tracks 02 & 03: 대칭 샴 DeBERTa-v3 (`12L` / `24L`)

### 백본 및 대칭 헤드

- **토큰 예산**: 문서당 최대 `512`토큰(쌍 합산 `1,024`토큰), 공유 DeBERTa-v3 인코더(`FP32` 마스터 가중치 + `BF16` autocast).
- **상위 50% 학습 가능 층 가중 풀링**: `Base` 상위 6층(`L7~L12`), `Large` 상위 12층(`L13~L24`)을 softmax 가중 결합.
- **어텐션 통계 풀링 (`ASP`)**: 토큰별 어텐션 가중 평균과 표준편차를 결합(`2 * H`)해 문체 호흡 변동성 반영.
- **대칭 상호작용 헤드**: `[0.5*(u+v), |u-v|, u_unit*v_unit, (u_unit-v_unit)^2, cos(u_unit, v_unit)]`를 LayerNorm-GELU MLP와 5 branch Multi-Sample Dropout(`p = 0.10..0.30`)으로 분류.

### 24층 Multi Std + Dual-Rank `All but the Top` 잔차 OOD 게이트 (Track 03)

도메인 라벨과 ML 피처 없이 제로샷 장르의 임베딩 이방성(Anisotropy)을 제거합니다.
1. 24개 층에서 4개 언어 깊이 추출:
   - **Band 1 (`L2~L6`)**: 어휘·구두점·철자 습관 (평균 + 표준편차 `s1`)
   - **Band 2 (`L8~L12`)**: 구문·기능어 리듬 (평균 + 표준편차 `s2`)
   - **Band 3 (`L14~L18`)**: 담화 전개 패턴 (평균 + 표준편차 `s3`)
   - **Band 4 (`L22~L24`)**: 고수준 의미·주제 표현 (평균)
2. 256개 고정 학습 사칭자에서 SVD로 **Dual-Rank `All but the Top`** (`r=3` 완만한 주제 제거, `r=8` 강한 주제 제거)을 적합하고 모든 깊이에서 대칭 `AS Norm`·Top 10 마진을 계산.
3. 71개 스칼라 기하·깊이 피처에 36차원 중심점 PCA(`0.5*(e1+e2)`), 원본 로짓 교차항 10개, 16차원 **대칭 합의축 PCA**(`e1 * e2` on `r=8`)를 결합.
4. 5 Bucket 양방향 저자 배제 OOF `0.35 LR(C=0.06) + 0.65 5 Seed ResidualOODGate`로 학습 후 `scikit-learn` 의존성이 없는 순수 텐서 파일(`model_03_deberta_v3_large.pt`)로 직렬화.
