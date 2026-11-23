# 설계 판단, 단계별 개선 과정 및 트러블슈팅

[English](./methodology-and-decisions.md) | [README](../README.ko.md)

각 모델, 특징 계층, 학습 설정, 사후 보정 기법을 **왜 선택했는지**, **어떤 가설을 왜 기각했는지**, 그리고 Locked Test를 단 1회 평가하기 전까지 Validation 진단이 어떤 개선으로 이어졌는지를 정리한 문서입니다.

---

## 1. 벤치마크 및 평가 프로토콜 설계 판단

| 설계 결정 | 선택 이유 | 기각한 대안 |
| :--- | :--- | :--- |
| **양방향 저자 완전 격리 (`benchmark_1024`)** | 기존 분할은 학습과 평가에 동일 문서·저자가 섞여 문서 암기로 점수가 과대평가됨. Train/Val/Test 간 저자 중복 0명으로 실제 저자 판별력만 측정. | 무작위 쌍 분할 또는 저자 ID 없는 문서 분할. |
| **5-Bucket 양방향 저자 배제 OOF** | `GroupKFold(groups=author_1)`는 `author_2`가 폴드 간에 누수됨. `(b(a1) != k) & (b(a2) != k)`로 Stage 1과 메타 보정기의 저자 누수를 차단. | 단방향 `GroupKFold` 또는 `StratifiedKFold`. |
| **순수 ML vs 순수 DL 분리 원칙** | 두 패러다임의 피처·예측값을 혼합하지 않아 딥러닝 스케일링 이득과 전통 문체학의 잔여 우위 구간(`cross_genre`)을 공정하게 규명. | ML 피처와 DeBERTa 로짓의 하이브리드 블렌딩. |
| **귀납적 단일 쌍 추론** | 테스트 배치 전체 통계를 쓰는 변환적 정규화는 실시간 2문장 데모에서 재현 불가. 256 고정 사칭자와 앵커 자체 사전값만 사용해 라이브 데모와 정합성 일치. | 테스트 배치 Z-정규화 또는 변환적 CSLS. |

---

## 2. Track 01 (`Stylometric ML`): 3라운드 단계별 개선

| 단계 | Val Closed Top-1 | Val Open ROC AUC | Val PAN Overall | 핵심 진단 및 개선 조치 |
| :--- | :---: | :---: | :---: | :--- |
| 기존 단일 표면 LightGBM | 48.67% | 0.7632 | 0.7336 | 표면 n gram이 동일 주제 어휘에 과적합. |
| 1차: 6계층 문체 이종 앙상블 | 60.00% | 0.8009 | 0.7520 | `POS-Noise`/`Distorted` 3중 뷰, 256 사칭자 `AS-Norm`, 64-Bag Koppel, Gram2Vec, NCD 통합. |
| 2차: 온도 정렬 8전문가 + 가지치기 | 62.00% | 0.8005 | 0.7594 | 미보정 LightGBM 로짓 진폭(`±6`)으로 앙상블 가중치가 `0.005`로 붕괴한 원인 규명 → 폴드 내부 온도 보정 + Gain 0 피처 `1,702`개 제거(`6,001 → 1,500`) + 12종 교차 피처. |
| **3차: 10전문가 + OOD 레지스터 게이트 (`d=0.068`)** | **67.33%** | **0.8378** | **0.7929** | `Guardian`(`89.6%`)·`ArXiv`(`75.4%`) 오탐 폭주의 원인이 구어체 사칭자 대비 격식체 레지스터 착시임을 확인 → 도메인 라벨 없이 `Genre-Blind CatBoost(642열)`, 교차 주제 `2.5x` LGBM, 내재적 OOD 레지스터 게이트 도입. |
| **Locked Test (고정 임계값)** | **64.58%** | **0.8474** | **0.8008** | 두 제로샷 도메인 Test FPR 모두 `25.7%`로 감소, Val 대비 AUC `+0.96%p` 상승. |

**Track 01 시스템 트러블슈팅:**
- **OpenBLAS / OpenMP 스레드 경합 (`15분 → 0.5초`)**: 트리 학습 직후 `scikit-learn` L-BFGS 스레드 충돌을 `threadpool_limits(limits=1)`로 격리.
- **크로스 버전 포터블 직렬화 (`21.63 MB`)**: NumPy 2.0/1.26 피클 호환 오류를 막기 위해 순수 Python 내장 타입으로 직렬화하고, `245.8 MB` Koppel 투영 행렬을 `150 Byte` `PCG64` 시드 상태로 압축(복원 오차 `0.000000`).

---

## 3. Track 02 (`DeBERTa-v3-base`): 트러블슈팅, 4단계 절제 실험 & 4차 OOD 보정

### 원인 규명: `transformers 5.x` 환경의 `FP16` 언더플로
초기 `DeBERTa-v3-base` 학습이 `loss = 0.6965`(`Val AUC = 0.5152`)에서 멈춘 현상을 추적해, `transformers 5.x`가 Hub에 `float16`으로 저장된 DeBERTa-v3 가중치를 그대로 FP16으로 적재하면서 `AdamW(1.6e-5)` 갱신이 FP16 기계 입실론(`~1e-3`) 아래로 반올림 소멸함을 밝혔습니다. 백본 로드 직후 `.float()`로 FP32 마스터 가중치를 강제해 정상 수렴을 복구했습니다.

### 4단계 통제 절제 실험 (`40%` 데이터 `33,944`쌍 x `1.5` epoch)

| 단계 | Val Closed Top-1 | Val Open ROC AUC | Val PAN Overall | 판정 및 원인 |
| :--- | :---: | :---: | :---: | :--- |
| `Step 0`: 기본 대칭 샴 베이스라인 | 67.33% | 0.8005 | 0.7690 | 기준선. |
| `Step 1`: + 상위 6층 + ASP + MSD + 15% EntityMask | 66.67% | **0.8083** | **0.7697** | **채택**: Open AUC `+0.78%p` 및 Closed Macro F1 개선. |
| `Step 2`: + PriorWD + LLRD + CoSENT (`shuffle=False`) | 60.67% | 0.7927 | 0.7547 | **기각**: `CoSENT`용 도메인순 정렬(`shuffle=False`)이 앞 도메인 망각을 유발. |
| `Step 3`: + Greedy Soup + EMA + AS-Norm + CSLS | 62.00% | **0.8098** | **0.7785** | **채택(S1 분리)**: Open PAN Overall `+0.95%p`; Closed 1:30은 원시 코사인 강제 결합을 분리해 Validation 최적 로짓 자동 선택. |

### 4단계 순수 신경망 사후 고도화 (재학습 0초)
전체 2.5 epoch(`shuffle=True`) 학습 직후 기존 도메인은 우수했으나 제로샷 장르 2곳(`Guardian FPR 97.6%`, `ArXiv FPR 94.8%`)의 임베딩 뭉침으로 전체 Val AUC가 `0.8155`에 머물렀습니다.
1. **2차 (`02c`: Neural OOD-Manifold Gate)**: 256 사칭자 Top-10 친화도 + L2 노름 + `PCA(24)` + PyTorch 잔차 MLP → Val AUC **`0.8155 → 0.8912` (`+7.57%p`)**, 제로샷 FPR `-60~67%p` 급감.
2. **3차 (`02e`: 중간 구문층 `L4~L6` + 중심화)**: 구문층과 최상위 의미층 괴리도 도입 → Val AUC **`0.8956`**, Closed Top-1 `76.00%`.
3. **4차 (`02f`: 4계층 + `All-but-the-Top r=3`)**: 4개 언어 깊이에서 사칭자 상위 주제 3축 직교 제거 → Val AUC **`0.9009`**, Closed Top-1 **`77.33%`**, Locked Test **`0.9040` / `76.25%`**.

---

## 4. Track 03 (`DeBERTa-v3-large`): 4-Way HP 프로브, 지연 스냅샷 수프 & Dual-Rank ABTT

### 24층에서 `LLRD + Prior WD + Pure BCE`를 선택한 근거
`shuffle=True` 조건에서 10% 층화 서브셋(`8,486`쌍) 4-Way 프로브를 수행해 Base `Step 2` 하락의 원인이 `LLRD`/`Prior WD`가 아니라 `shuffle=False`였음을 입증했습니다.
- `Probe_B (lr=1.6e-5 / 1.0e-4, LLRD gamma=0.94868, Prior WD 0.01, Pure BCE)`: **Val Composite `0.71345`** (`Closed Top-1 62.67%`, `Open AUC 0.78423`) / 채택.
- `Probe_A (1.2e-5, LLRD + Prior WD, Pure BCE)`: `0.69180`.
- `Probe_C (1.2e-5, LLRD + Prior WD, CoSENT 0.08)`: `0.69024` / 보조 CoSENT 손실이 순수 BCE 대비 이득 없음 확인.
- `Probe_D (1.0e-5, 균일 학습률)`: `0.67642` / LLRD+Prior WD 대비 `-3.70%p` 열세.

### L4 24GB VRAM 관리 및 최종 고도화 (`0.65/0.35` 수프 + Dual-Rank ABTT + AgreePCA)
- **지연 스냅샷 평가(Deferred Snapshot Evaluation)**: 24층 학습 도중 옵티마이저·Prior WD 앵커가 GPU에 상주한 상태에서 검증 추론을 실행하면 `L4 22.03 GB` 한계를 초과했습니다. 스냅샷(`50%, 62.5%, 75%, 87.5%, 100%`)을 CPU 디스크에 저장하고 학습 종료 후 옵티마이저를 해제한 뒤 일괄 평가하는 구조로 OOM을 완전히 해소했습니다.
- **최종 채택 구성 (`0.65 * step_4141 + 0.35 * step_4970` + Dual-Rank ABTT `r=3, 8` + Multi-Std `s1,s2,s3` + 대칭 합의축 PCA `e1*e2`)**:
  - `R2` 기준선(`step_4141 + 4-Depth r=3`): Val Open AUC `0.91009`, `ArXiv 0.8998`, `Guardian 0.7642`.
  - Dual-Rank ABTT(`r=3`과 `r=8`의 주제 제거 민감도 격차 `an_r8 - an_r3`), 3개 하위 깊이 토큰 표준편차(`s1/s2/s3`), 대칭 합의축 PCA(`e1 * e2`, 16d)를 추가해 Val AUC `0.91237`, `ArXiv 0.9080`으로 향상.
  - 상위 2개 스냅샷 선형 보간 수프(`0.65 / 0.35`)와 확장 귀납적 CSLS(`hw=1.15`)를 적용해 **Val Open AUC `0.91388`**, **Locked Test Open AUC `0.91393`**(Val vs Test 격차 `0.00005`), **Closed Top-1 `78.33%`**, <strong>제로샷 <code>ArXiv</code> AUC <code>0.9389</code> (<code>FPR 8.6%</code>)</strong>를 달성했습니다.
