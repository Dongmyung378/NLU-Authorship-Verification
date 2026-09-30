# 저자 동일성 판별

**Dongmyung Park과 Juho Kim의 공동 NLP 프로젝트**

[English](./README.md) · [재현 가이드](./docs/reproduction.ko.md) · [프로젝트 포스터](./docs/authorship-verification-poster.pdf)

두 영어 글이 같은 사람의 글인지 판별할 수 있을까요? 이 프로젝트는 가벼운 문체 기반 앙상블과 파인튜닝한 RoBERTa-Large, 두 개의 완전한 분류 파이프라인으로 그 문제를 탐구합니다. 저장소에는 데이터 분할, 학습·추론 노트북, 저장된 예측 결과, 모델 문서, 두 모델을 직접 시험할 수 있는 터미널 시연 도구가 들어 있습니다.

## 주요 결과

| 모델 | 방법 | Macro F1 | 정확도 | 실용적 특징 |
|---|---|---:|---:|---|
| [문체 기반 앙상블](./model-cards/stylometric-ensemble.ko.md) | 단어·문자·기능어 TF-IDF, 문체 특징 24개, XGBoost + LightGBM | 0.7702 | 0.77 | CPU 추론 가능, 모델 파일 60MB 미만 |
| [RoBERTa-Large + ASL](./model-cards/roberta-asymmetric-loss.ko.md) | Asymmetric Loss를 사용한 텍스트 쌍 Transformer | **0.8348** | **0.83** | 더 높은 개발 세트 점수, 약 1.32GB의 가중치 |

두 점수는 동일한 **라벨이 있는 개발 세트 5,993쌍**(서로 다른 저자 2,937쌍, 동일 저자 3,056쌍)에서 얻었습니다. 분류 임계값도 이 개발 세트에서 선택했으므로, 표의 값은 별도의 미사용 최종 테스트 결과가 아니라 기록된 개발 단계 평가 결과입니다. 저장소의 테스트 세트 5,985쌍에는 라벨이 없어 테스트 정확도나 F1은 제시하지 않습니다.

## 구현 내용

**문체 기반 파이프라인**은 저자를 드러낼 수 있는 메타데이터를 정리한 뒤 단어·문자 n-gram, 기능어 사용 패턴, 문장 길이·구두점·어휘 다양성 등을 다루는 문체 특징 24개로 각 글을 표현합니다. 두 글의 특징을 조합해 XGBoost와 LightGBM에 입력하고, 조정된 가중치와 `0.43`의 분류 임계값으로 결과를 결합합니다. 로컬 CPU에서 사용하기 좋은 작은 모델입니다.

**RoBERTa 파이프라인**은 이메일 헤더·주소·URL을 정리하고 두 글을 합쳐 최대 512토큰으로 인코딩합니다. `roberta-large`를 Asymmetric Loss(`gamma_neg=2`, `gamma_pos=1`)로 파인튜닝했으며, 개발 세트에서 선택한 임계값은 `0.51`입니다. 맥락 정보를 활용하는 대신 연산량과 저장 용량이 더 큽니다.

두 파이프라인 모두 학습·추론 노트북과 모델 카드를 제공합니다. [`verify_author.py`](./verify_author.py)는 두 모델에 공통으로 사용할 수 있는 단일 텍스트 쌍 명령행 인터페이스입니다.

## 직접 예측해 보기

### 1. Python 환경 구성

Python 3.12를 권장합니다. 시스템에서 사용할 수 있는 Python 3.12 실행 명령으로 가상 환경을 만든 뒤, 터미널 추론에 필요한 패키지를 설치하세요. Windows PowerShell 예시는 다음과 같습니다.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-inference.txt
```

`python` 명령을 사용할 수 없다면 설치된 다른 Python 실행 명령으로 `.venv`를 만드세요. 가상 환경을 만든 후에는 위와 아래의 명령이 그 안의 Python을 직접 실행합니다. macOS 또는 Linux에서는 `.\.venv\Scripts\python.exe` 대신 `.venv/bin/python`을 사용합니다. 노트북 실행과 재학습에는 전체 [`requirements.txt`](./requirements.txt)를 설치하세요. 환경 구성과 CUDA 안내는 [재현 가이드](./docs/reproduction.ko.md)에 있습니다.

**Windows에서 NVIDIA GPU로 RoBERTa를 실행하려면** 위 의존성 설치 후(노트북도 사용할 경우 `requirements.txt` 설치 후) CUDA 12.4용 PyTorch를 설치하세요.

```powershell
.\.venv\Scripts\python.exe -m pip install --force-reinstall --no-deps torch==2.5.1+cu124 --index-url https://download.pytorch.org/whl/cu124
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No CUDA device')"
```

확인 결과에서 CUDA 사용 가능 여부가 `True`이고 GPU 이름이 표시되어야 합니다. CUDA 12.4가 적합하지 않으면 [PyTorch 공식 버전 목록](https://pytorch.org/get-started/previous-versions/)에서 운영체제와 NVIDIA 드라이버에 맞는 빌드를 선택하세요. **Windows의 Git Bash**에서는 아래 시연 명령을 포함해 `./.venv/Scripts/python.exe` 경로를 사용합니다.

### 2. 모델 다운로드

학습된 모델 파일은 크기 때문에 GitHub 밖에 보관합니다. 다운로드 링크는 다음과 같습니다.

| 모델 | 다운로드 | 파일을 바로 넣을 수 있는 폴더 |
|---|---|---|
| 문체 기반 앙상블 | [Google Drive: XGBoost, LightGBM, 벡터라이저](https://drive.google.com/drive/folders/1mF0bMpzfs3XODpQA2Hd7H_JvXLk5AGgp?usp=sharing) | `models/stylometric-ensemble/` 또는 `models/Models_ML/` |
| RoBERTa-Large + ASL | [Google Drive: 파인튜닝 체크포인트와 토크나이저](https://drive.google.com/drive/folders/1Ty9-EpOLAgzvTWPuqrHBTbehmZZuDs3V?usp=sharing) | `models/roberta-asymmetric-loss/` 또는 `models/Models_ASL/` |

Google Drive에서 내려받은 `Models_ML`, `Models_ASL` 폴더는 이름을 바꾸지 않아도 터미널 시연에 사용할 수 있습니다. 해당 폴더를 `models/` 안에 넣고, 모델 파일이 추가 하위 폴더 없이 바로 그 안에 있도록 배치하세요. 노트북은 `AV_MODEL_DIR`을 지정하지 않으면 위 표의 표준 폴더(`stylometric-ensemble`, `roberta-asymmetric-loss`)를 사용합니다. `models/`는 Git에서 제외되어 다운로드한 가중치는 로컬에만 남습니다.

터미널 시연 도구가 확인하는 파일은 다음과 같습니다.

```text
models/Models_ML/                 # 또는 models/stylometric-ensemble/
├── ensemble_config.json
├── lgb_model.pkl
├── tfidf_char.pkl
├── tfidf_fw.pkl
├── tfidf_word.pkl
└── xgb_model.json

models/Models_ASL/                # 또는 models/roberta-asymmetric-loss/
├── best_threshold.json
├── config.json
├── model.safetensors
├── tokenizer.json
└── tokenizer_config.json
```

RoBERTa 다운로드에 다른 토크나이저 파일도 포함되어 있다면 같은 폴더에 보관하세요. 실행할 모델의 파일만 다운로드해도 됩니다.

### 3. 터미널 시연 실행

저장소 루트에서 실행해 영어 글 두 개를 입력합니다. 기본 모델은 문체 기반 앙상블입니다.

```powershell
.\.venv\Scripts\python.exe verify_author.py
```

GPU에서 RoBERTa를 실행하거나 입력 글을 명령행 인자로 전달할 수도 있습니다.

```powershell
.\.venv\Scripts\python.exe verify_author.py --model roberta --device cuda
.\.venv\Scripts\python.exe verify_author.py --text-1 "I walked by the river this morning and watched the boats pass." --text-2 "The rain started quietly in the evening, so I closed the windows."
```

**Windows Git Bash**에서 실행할 때는 다음 명령을 사용합니다.

```bash
./.venv/Scripts/python.exe verify_author.py --model roberta --device cuda
```

실행 결과에는 **Same author**(동일 저자) 또는 **Different authors**(다른 저자), 동일 저자 확률, 분류 임계값이 표시됩니다. 모델 파일이 다른 위치에 있다면 `--model-dir PATH`를 사용하세요. RoBERTa의 기본값 `--device auto`는 CUDA를 사용할 수 있으면 GPU를 사용합니다. NVIDIA GPU를 감지했지만 PyTorch에서 CUDA를 사용할 수 없다면 CPU로 조용히 전환하지 않고 설정 방법을 안내하며 중단합니다. `--device cuda`는 CUDA를 필수로 요구하며, 느린 CPU 추론을 의도적으로 실행할 때만 `--device cpu`를 지정하세요. 문체 앙상블은 CPU에서 실행됩니다.

## 저장소 구성

```text
data/                       라벨이 있는 학습·개발 쌍과 라벨 없는 테스트·예시 쌍
notebooks/                  두 파이프라인의 학습·시연 노트북
model-cards/                구조, 하이퍼파라미터, 평가, 한계
models/                     다운로드한 모델 파일(Git 추적 제외)
results/                    라벨 없는 테스트 세트의 저장된 예측 결과
docs/                       재현 가이드와 프로젝트 포스터
verify_author.py            대화형·인자형 터미널 시연 도구
requirements-inference.txt  터미널 시연용 의존성
requirements.txt            노트북·학습용 전체 의존성
```

라벨이 있는 [`train.csv`](./data/train.csv)는 27,643쌍, [`dev.csv`](./data/dev.csv)는 5,993쌍입니다. 라벨이 없는 [`test.csv`](./data/test.csv)와 [`live.csv`](./data/live.csv)는 각각 5,985쌍, 20쌍입니다. 각 쌍에는 `text_1`, `text_2`가 있으며, `label=1`은 동일 저자, `label=0`은 다른 저자를 뜻합니다. 노트북은 저장소에서 그대로 볼 수 있으며, [`.gitattributes`](./.gitattributes)가 `.ipynb` 파일을 GitHub 언어 통계에서 제외합니다.

## 자세히 보기

- [재현 가이드](./docs/reproduction.ko.md): 로컬 환경, 모델 파일 배치, 추론, 재학습.
- [문체 기반 모델 카드](./model-cards/stylometric-ensemble.ko.md)와 [RoBERTa 모델 카드](./model-cards/roberta-asymmetric-loss.ko.md): 방법, 하이퍼파라미터, 연산 환경, 결과, 위험 요소.
- [학습·시연 노트북](./notebooks/): 전체 실험과 배치 추론.
- [프로젝트 포스터](./docs/authorship-verification-poster.pdf): 프로젝트 시각 요약.

## 한계

두 방법 모두 영어 텍스트를 대상으로 하며, 매우 짧은 글·다른 언어·코드 스위칭·낯선 장르에서는 결과가 불안정할 수 있습니다. RoBERTa는 두 글의 입력이 512토큰을 넘으면 일부가 잘리고, 문체 모델은 학습된 TF-IDF 어휘 밖의 표현을 담지 못합니다. 성능 수치는 이 개발 세트에 한정됩니다. 모델 점수는 문체 패턴에 대한 단서일 뿐 신원을 증명하지 않으므로, 저자 귀속·제재·익명 해제의 단독 근거로 사용하면 안 됩니다.

## 개발자

Dongmyung Park과 Juho Kim이 함께 개발했습니다.
