# 재현 가이드

[English](./reproduction.md) · [프로젝트 개요](../README.ko.md)

이 문서는 두 저자 동일성 판별 파이프라인의 로컬 환경 구성, 모델 파일 배치, 평가, 배치 추론, 선택적 재학습 방법을 설명합니다.

## 1. 실행 환경

추론 노트북은 Python 3.12.7에서 검증했습니다. 저장소 루트에서 격리 환경을 생성합니다.

~~~bash
python -m venv .venv

# PowerShell
.\.venv\Scripts\Activate.ps1

# macOS 또는 Linux
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
jupyter lab
~~~

고정된 <code>torch</code> 패키지는 CPU 추론을 지원합니다. NVIDIA GPU를 사용하려면 나머지 의존성을 설치하기 전에 시스템의 CUDA 런타임과 호환되는 PyTorch 빌드를 설치하세요. 학습에 사용한 라이브러리 버전은 각 [모델 카드](../model-cards/)에 기록되어 있습니다.

## 2. 모델 파일 다운로드

RoBERTa 체크포인트의 크기가 약 1.32GB이므로 학습된 모델 파일은 별도로 제공합니다.

### 문체 기반 앙상블

[XGBoost, LightGBM 및 벡터라이저 파일을 다운로드](https://drive.google.com/drive/folders/1mF0bMpzfs3XODpQA2Hd7H_JvXLk5AGgp?usp=sharing)한 뒤 다음과 같이 배치합니다.

~~~text
models/stylometric-ensemble/
├── ensemble_config.json
├── lgb_model.pkl
├── tfidf_char.pkl
├── tfidf_fw.pkl
├── tfidf_word.pkl
└── xgb_model.json
~~~

### RoBERTa-Large

[파인튜닝된 RoBERTa 파일을 다운로드](https://drive.google.com/drive/folders/1Ty9-EpOLAgzvTWPuqrHBTbehmZZuDs3V?usp=sharing)한 뒤 다음과 같이 배치합니다.

~~~text
models/roberta-asymmetric-loss/
├── best_threshold.json
├── config.json
├── model.safetensors
├── tokenizer.json
└── tokenizer_config.json
~~~

토크나이저 폴더에 어휘 파일이 추가로 포함되어 있을 수 있습니다. 내려받은 파일은 모두 같은 디렉터리에 보관하세요.

## 3. 데이터 구성

노트북은 저장소의 <code>data/</code> 디렉터리에서 데이터셋을 읽습니다.

~~~text
data/
├── train.csv
├── dev.csv
├── test.csv
└── live.csv
~~~

필수 열:

- <code>text_1</code>, <code>text_2</code>: 입력 텍스트
- <code>label</code>: 학습·개발 데이터에 포함되며 동일 저자는 <code>1</code>, 다른 저자는 <code>0</code>

## 4. 추론 실행

저장소 루트에서 Jupyter를 시작한 뒤 다음 노트북 중 하나를 엽니다.

- [문체 기반 앙상블 데모](../notebooks/stylometric-ensemble/demo.ipynb)
- [RoBERTa-Large 데모](../notebooks/roberta-asymmetric-loss/demo.ipynb)

셀을 순서대로 실행하면 라벨이 있는 개발 데이터를 평가하고, <code>test.csv</code>의 예측을 생성해 <code>results/</code>에 저장하며, 단일 텍스트 쌍을 예측하는 함수도 사용할 수 있습니다.

문체 기반 모델은 CPU 추론을 지원합니다. RoBERTa-Large를 실용적인 속도로 실행하려면 CUDA를 지원하는 GPU를 권장합니다.

## 5. 모델 재학습

학습 노트북:

- [문체 기반 앙상블 학습](../notebooks/stylometric-ensemble/training.ipynb)
- [RoBERTa-Large 학습](../notebooks/roberta-asymmetric-loss/training.ipynb)

기본 출력 폴더:

- <code>models/stylometric-ensemble/</code>
- <code>models/roberta-asymmetric-loss/</code>

학습에는 추론보다 훨씬 많은 연산 자원이 필요합니다. 기록된 실행에서는 부스팅 앙상블에 NVIDIA T4 GPU 2개를, RoBERTa-Large에 NVIDIA A100 40GB GPU 1개를 사용했습니다. 사용 가능한 하드웨어에 맞게 배치 크기와 장치 설정을 조정하세요.

## 6. 경로 재정의

노트북은 저장소 루트를 자동으로 찾습니다. 호스팅 노트북 환경이나 별도 스토리지를 사용할 때는 다음 환경 변수로 경로를 재정의할 수 있습니다.

| 환경 변수 | 용도 |
|---|---|
| <code>AV_PROJECT_ROOT</code> | 저장소 루트 |
| <code>AV_DATA_DIR</code> | 데이터셋 디렉터리 |
| <code>AV_MODEL_DIR</code> | 모델 입력 또는 출력 디렉터리 |
| <code>AV_RESULTS_DIR</code> | 예측 결과 디렉터리 |

설치된 패키지나 환경 변수를 변경한 뒤에는 Jupyter 커널을 다시 시작하세요.

## 7. 예상 출력

배치 추론은 <code>prediction</code> 열 하나로 구성된 CSV 파일을 생성합니다.

~~~text
results/
├── stylometric-ensemble.csv
└── roberta-asymmetric-loss.csv
~~~

각 파일에는 <code>data/test.csv</code>의 행 순서와 일치하는 5,985개의 예측이 저장됩니다.
