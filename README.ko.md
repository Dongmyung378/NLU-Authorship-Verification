# 저자 동일성 판별

[English](./README.md) · [재현 가이드](./docs/reproduction.ko.md) · [문체 앙상블 모델 카드](./model-cards/stylometric-ensemble.ko.md) · [RoBERTa 모델 카드](./model-cards/roberta-asymmetric-loss.ko.md) · [프로젝트 포스터](./docs/authorship-verification-poster.pdf)

두 영어 글이 같은 사람에 의해 작성되었는지 예측하는 저자 동일성 판별 프로젝트입니다. 해석 가능한 문체 기반 앙상블과 RoBERTa-Large 파인튜닝 모델을 비교해, 효율적인 추론과 높은 예측 성능 사이의 차이를 보여줍니다.

## 핵심 성과

- RoBERTa-Large와 Asymmetric Loss로 **Macro F1 0.8348** 달성
- CPU에서도 실행 가능한 XGBoost-LightGBM 문체 앙상블로 **Macro F1 0.7702** 달성
- 전처리, 학습, 임계값 최적화, 평가, 배치 추론, 대화형 예측을 포함한 전체 노트북 제공
- 구조, 하이퍼파라미터, 연산 환경, 한계와 책임 있는 사용법을 모델 카드에 정리

## 결과

두 모델은 동일한 5,993개 균형 개발 텍스트 쌍에서 평가했습니다.

| 모델 | 접근 방식 | Macro F1 | 정확도 | 추론 특성 |
|---|---|---:|---:|---|
| 문체 기반 앙상블 | 단어·문자·기능어 TF-IDF, 문체 특징 24개, XGBoost + LightGBM | 0.7702 | 0.77 | CPU 추론 가능, 모델 파일 60MB 미만 |
| RoBERTa + ASL | 비대칭 초점 패널티를 적용한 RoBERTa-Large 텍스트 쌍 분류 | **0.8348** | **0.83** | GPU 권장, 가중치 약 1.32GB |

표의 수치는 모델 카드에 기록된 저장 평가 결과를 기준으로 합니다.

## 모델 구성

### 문체 기반 앙상블

명백한 메타데이터를 정규화한 뒤 단어·문자 n-gram, 기능어 사용 패턴, 문장 길이, 구두점 비율, 어휘 다양성, 접미사 패턴 등 24개의 문체 특징을 추출합니다. 두 글의 차이·곱·비율·코사인 거리를 XGBoost와 LightGBM에 입력하고, 두 모델의 확률을 최적화된 가중치와 임계값으로 결합합니다.

### Asymmetric Loss를 적용한 RoBERTa-Large

이메일 헤더, 주소, URL, 중복 공백을 정리한 뒤 각 텍스트 쌍을 최대 512토큰으로 변환합니다. 구분하기 어려운 다른 저자 쌍에 더 집중하도록 Asymmetric Loss(`gamma_neg=2`, `gamma_pos=1`)로 RoBERTa-Large를 파인튜닝했습니다. 개발 세트 임계값 탐색을 통해 최종 분류 임계값 `0.51`을 선택했습니다.

## 데이터셋

| 분할 | 텍스트 쌍 | 라벨 | 용도 |
|---|---:|---|---|
| `train.csv` | 27,643 | 있음 | 모델 학습 |
| `dev.csv` | 5,993 | 있음 | 평가 및 임계값 최적화 |
| `test.csv` | 5,985 | 없음 | 배치 예측 |
| `live.csv` | 20 | 없음 | 소규모 추론 예시 |

각 행은 `text_1`, `text_2`를 포함하며, 라벨이 있는 분할에는 동일 저자 `1`, 다른 저자 `0`을 나타내는 `label` 열이 있습니다.

## 저장소 구조

```text
.
├── data/                         # 학습·개발·테스트·예시 데이터
├── notebooks/
│   ├── stylometric-ensemble/     # 문체 모델 학습 및 추론
│   └── roberta-asymmetric-loss/  # RoBERTa 모델 학습 및 추론
├── model-cards/                  # 상세 모델 문서
├── models/                       # 내려받은 가중치(Git 추적 제외)
├── results/                      # 저장된 테스트 예측
├── docs/                         # 재현 가이드와 프로젝트 포스터
├── requirements.txt
├── README.md                     # 영문 메인 문서
└── README.ko.md                  # 한국어 문서
```

## 빠른 시작

Python 3.12 사용을 권장합니다.

```bash
git clone https://github.com/Dongmyung378/NLU-Authorship-Verification.git
cd NLU-Authorship-Verification
python -m venv .venv

# PowerShell
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
jupyter lab
```

학습된 가중치를 내려받아 `models/stylometric-ensemble/` 또는 `models/roberta-asymmetric-loss/` 아래에 배치한 뒤 해당 데모 노트북을 실행합니다. 다운로드 링크, 필요한 파일, CUDA 안내, 경로 재정의 방법은 [재현 가이드](./docs/reproduction.ko.md)에 정리되어 있습니다.

## 한계와 책임 있는 사용

- 영어 텍스트에 최적화되어 있어 다국어, 코드 스위칭, 매우 짧은 글, 학습 분포 밖의 글에서는 성능이 떨어질 수 있습니다.
- RoBERTa는 두 입력의 합이 512토큰을 넘으면 뒷부분을 자르며, 문체 모델은 TF-IDF 학습 어휘에 없는 표현을 반영하지 못합니다.
- 성능 수치는 하나의 개발 데이터셋에 대한 결과이며 모든 환경의 성능을 보장하지 않습니다.
- 저자 판별 결과는 민감하게 사용될 수 있습니다. 신원 확인, 저자 귀속, 제재, 익명 해제의 단독 근거로 사용하지 마세요.

## 기여자

- Dongmyung Park
- Juho Kim
