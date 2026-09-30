# 재현 가이드

[English](./reproduction.md) · [프로젝트 개요](../README.ko.md)

제공된 영어 글 쌍 개발 세트에서 모델 결과를 재현하고, 터미널에서 다른 글에 대한 모델 출력을 시험하는 방법입니다. 보고된 성능은 이 개발 세트에 한정되며 새로운 저자나 도메인의 정확도는 측정되지 않았습니다. 학습된 모델 파일은 GitHub 저장소에 넣기에는 용량이 커서 별도로 내려받아야 합니다.

## 1. Python 환경 설정

Python 3.12를 사용하세요. 노트북은 3.12.7에서 확인했습니다. 다음 명령은 저장소 루트에서 실행합니다. 가상 환경을 활성화하지 않아도 실행 파일을 직접 호출할 수 있습니다.

**Windows PowerShell**

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-inference.txt
```

`py` 명령을 사용할 수 없다면 환경을 만들 때 `python`을 사용하세요. Jupyter 노트북 또는 재학습에도 사용할 경우 전체 의존성을 설치합니다.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

**macOS 또는 Linux**

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-inference.txt
# Jupyter 노트북 또는 재학습에도 사용할 경우:
.venv/bin/python -m pip install -r requirements.txt
```

`requirements-inference.txt`는 터미널 도구에 필요한 의존성을 포함합니다. `requirements.txt`에는 Jupyter 및 학습 의존성도 들어 있습니다.

**NVIDIA GPU를 사용하는 Windows:** 위 의존성을 설치한 뒤 CUDA 12.4용 PyTorch를 설치하세요. 노트북용 `requirements.txt`도 설치했다면 다음 명령을 그 뒤에 실행합니다.

```powershell
.\.venv\Scripts\python.exe -m pip install --force-reinstall --no-deps torch==2.5.1+cu124 --index-url https://download.pytorch.org/whl/cu124
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No CUDA device')"
```

확인 결과에 `2.5.1+cu124`, `12.4`, `True`, GPU 이름이 표시되어야 합니다. CUDA 사용 가능 여부가 `False`이면 RoBERTa를 실행하기 전에 NVIDIA 드라이버와 설치된 PyTorch 빌드를 확인하세요. CUDA 12.4는 호환되는 Windows 환경을 위한 예시입니다. 다른 환경에서는 [PyTorch 공식 이전 버전 목록](https://pytorch.org/get-started/previous-versions/)에서 운영체제와 드라이버에 맞는 빌드를 선택하세요. `torch` 패키지가 설치됐다는 사실만으로 CUDA 지원이 보장되지는 않습니다. Windows의 Git Bash에서는 인터프리터 경로를 `./.venv/Scripts/python.exe`로 적습니다. 위의 `.\.venv\Scripts\python.exe` 표기는 PowerShell용입니다.

## 2. 모델 파일 다운로드

필요한 모델 하나만 받거나 둘 다 받을 수 있습니다. 각 모델의 파일을 아래 폴더 **바로 안에** 넣으세요. Google Drive에서 받은 바깥 폴더가 한 겹 더 들어가면 안 됩니다. `models/` 전체는 Git에서 무시하므로 저장소 복제본에는 포함되지 않습니다.

| 모델 | 다운로드 | 기본 폴더 |
|---|---|---|
| 문체 특징 기반 XGBoost + LightGBM 앙상블 | [모델 및 벡터라이저 파일](https://drive.google.com/drive/folders/1mF0bMpzfs3XODpQA2Hd7H_JvXLk5AGgp?usp=sharing) | `models/stylometric-ensemble/` |
| 파인튜닝된 RoBERTa-Large | [체크포인트 및 토크나이저 파일](https://drive.google.com/drive/folders/1Ty9-EpOLAgzvTWPuqrHBTbehmZZuDs3V?usp=sharing) | `models/roberta-asymmetric-loss/` |

최소한 다음 파일이 있어야 합니다.

```text
models/
├── stylometric-ensemble/
│   ├── ensemble_config.json
│   ├── lgb_model.pkl
│   ├── tfidf_char.pkl
│   ├── tfidf_fw.pkl
│   ├── tfidf_word.pkl
│   └── xgb_model.json
└── roberta-asymmetric-loss/
    ├── best_threshold.json
    ├── config.json
    ├── model.safetensors
    ├── tokenizer.json
    └── tokenizer_config.json
```

추가로 받은 토크나이저 또는 학습 파일도 같은 폴더에 보관하세요. RoBERTa 가중치 파일만 약 1.3GiB이므로 디스크 여유 공간을 확인하세요.

터미널 도구는 원래 다운로드 폴더 이름인 `models/Models_ML/`(앙상블)과 `models/Models_ASL/`(RoBERTa)도 인식합니다. 기본 폴더를 먼저 확인하고, 그다음 해당 원래 이름의 폴더에서 필수 파일이 모두 있는지 확인합니다. **노트북은 기본적으로 위의 기본 폴더만 사용합니다.** 원래 이름을 그대로 쓰려면 폴더를 기본 이름으로 바꾸거나, Jupyter를 시작하기 전에 `AV_MODEL_DIR`을 실제 모델 폴더로 지정하세요(5절).

## 3. 터미널에서 예측하기

저장소 루트에서 대화형 시연을 시작합니다. 영어 글 두 개를 차례로 입력하며, 기본 모델은 CPU에서 사용하기 좋은 문체 앙상블입니다.

```powershell
.\.venv\Scripts\python.exe verify_author.py
```

macOS 또는 Linux에서는 `.\.venv\Scripts\python.exe`를 `.venv/bin/python`으로 바꾸세요. RoBERTa를 선택하거나, 두 글을 명령행에 바로 입력하거나, `models/` 밖의 모델 폴더를 지정할 수도 있습니다.

```powershell
.\.venv\Scripts\python.exe verify_author.py --model roberta --device cuda
.\.venv\Scripts\python.exe verify_author.py --text-1 "I walked by the river this morning." --text-2 "The rain began quietly that evening."
.\.venv\Scripts\python.exe verify_author.py --model roberta --model-dir "D:\models\Models_ASL" --text-1 "First passage." --text-2 "Second passage."
```

Unix 경로라면 `--model-dir /path/to/Models_ASL`처럼 지정하세요. `--model-dir`은 상위 폴더가 아니라 위 파일이 직접 들어 있는 폴더를 가리켜야 합니다. `--text-1`과 `--text-2`는 함께 제공해야 하며, 둘 다 생략하면 입력을 요청합니다.

**Windows Git Bash**에서는 다음처럼 실행합니다.

```bash
./.venv/Scripts/python.exe verify_author.py --model roberta --device cuda
```

RoBERTa의 기본값인 `--device auto`는 PyTorch가 CUDA를 사용할 수 있으면 GPU를 선택합니다. NVIDIA GPU를 감지했지만 PyTorch에서 CUDA를 사용할 수 없다면 CPU로 조용히 전환하지 않고 설정 안내와 함께 중단합니다. `--device cuda`는 CUDA를 필수로 요구하며, `--device cpu`는 느린 CPU 추론을 명시적으로 선택합니다. 문체 모델은 CPU를 사용합니다.

출력에는 `Prediction`, `Same-author probability`, `Decision threshold`가 표시됩니다. 모델 점수가 저장된 임곗값 **이상**이면 “Same author”, 미만이면 “Different authors”로 판정합니다. 출력 이름과 달리 이 수치는 두 글이 같은 저자의 글일 실제 확률이 아니라 보정되지 않은 모델 점수입니다. 매우 짧은 글은 표시되는 점수가 높더라도 결과가 불안정할 수 있습니다.

점수와 임곗값은 백분율로 표시됩니다.

## 4. 노트북 실행 및 결과 확인

1절처럼 `requirements.txt`를 설치하고 저장소 루트에서 Jupyter를 시작합니다.

```powershell
.\.venv\Scripts\jupyter.exe lab
```

macOS 또는 Linux에서는 `.venv/bin/jupyter lab`을 실행하세요. [문체 앙상블 데모](../notebooks/stylometric-ensemble/demo.ipynb) 또는 [RoBERTa 데모](../notebooks/roberta-asymmetric-loss/demo.ipynb)를 열고 셀을 순서대로 실행합니다. 각 노트북은 다음 작업을 합니다.

1. 해당 모델을 불러와 정답이 있는 `data/dev.csv`를 평가합니다.
2. 정답이 없는 `data/test.csv`를 예측해 `results/` 아래에 한 열짜리 CSV를 저장합니다.
3. `predict_author_match` 함수를 정의하고 예시 글 쌍을 실행합니다. 마지막 셀의 `text_a`와 `text_b`를 바꾸면 직접 쓴 글도 넣을 수 있습니다.

결과 파일은 `results/stylometric-ensemble.csv`와 `results/roberta-asymmetric-loss.csv`입니다. 각각 `prediction` 열에 테스트 데이터의 원래 행 순서대로 0 또는 1이 한 개씩 기록됩니다. 현재 `test.csv`에는 5,985행이 있습니다. `1`은 동일 저자, `0`은 다른 저자를 뜻합니다. 테스트 데이터에는 정답 열이 없으므로 이 CSV는 측정된 테스트 점수가 아닙니다. 공개된 개발 지표는 임곗값과 RoBERTa의 최적 체크포인트를 고를 때에도 사용한 `dev.csv`에서 얻었습니다. 제공된 `train.csv`와 `dev.csv`를 직접 비교하면 **개발 세트 5,993쌍 중 630쌍(10.5%)이 학습 쌍과 완전히 동일**합니다. 따라서 이 지표는 새로운 저자나 도메인에서의 독립적인 성능을 입증하지 않습니다. 일반화 성능을 평가하려면 저자와 도메인을 분리한 별도의 라벨 있는 데이터가 필요합니다.

## 5. 데이터와 경로 변경

제공된 데이터는 `data/`에 있습니다.

| 파일 | 열 | 용도 |
|---|---|---|
| `train.csv` | `text_1`, `text_2`, `label` | 학습 |
| `dev.csv` | `text_1`, `text_2`, `label` | 개발 데이터 평가와 임곗값 선택 |
| `test.csv` | `text_1`, `text_2` | 정답 없는 일괄 추론 |
| `live.csv` | `text_1`, `text_2` | 추가 정답 없는 글 쌍 |

노트북은 현재 작업 폴더에서 상위로 올라가며 `data/`가 있는 저장소를 찾습니다. 파일을 다른 위치에 둔 경우 Jupyter를 시작하기 전에 다음 환경 변수를 설정하세요.

| 환경 변수 | 노트북에서 가리키는 경로 |
|---|---|
| `AV_PROJECT_ROOT` | 저장소 루트 |
| `AV_DATA_DIR` | CSV 파일이 있는 폴더 |
| `AV_MODEL_DIR` | **선택한 모델의 파일**이 있는 폴더 |
| `AV_RESULTS_DIR` | 예측 CSV를 저장할 폴더 |

예를 들어 Windows에서 문체 모델의 원래 폴더 이름을 유지하려면 다음과 같이 실행합니다.

```powershell
$env:AV_MODEL_DIR = (Resolve-Path .\models\Models_ML).Path
.\.venv\Scripts\jupyter.exe lab
```

macOS 또는 Linux에서는 `AV_MODEL_DIR="$PWD/models/Models_ML" .venv/bin/jupyter lab`을 사용하세요. RoBERTa 노트북을 실행할 때는 `Models_ML`을 `Models_ASL`로 바꾸세요. 터미널 도구는 `AV_MODEL_DIR` 대신 `--model-dir`을 사용합니다. 환경 변수나 설치 패키지를 변경했다면 Jupyter 커널을 다시 시작하세요.

## 6. 선택적 재학습과 하드웨어

[문체 앙상블 학습 노트북](../notebooks/stylometric-ensemble/training.ipynb)과 [RoBERTa 학습 노트북](../notebooks/roberta-asymmetric-loss/training.ipynb)은 `train.csv`와 `dev.csv`를 읽습니다. 기본적으로 각 모델의 기본 폴더에 결과를 저장하며, `AV_MODEL_DIR`로 출력 폴더를 바꿀 수 있습니다. RoBERTa 노트북은 기본 `roberta-large` 모델을 불러오므로 캐시에 없으면 별도로 다운로드해야 할 수 있습니다.

RoBERTa-Large 학습에는 추론보다 훨씬 많은 메모리와 연산 자원이 필요합니다. 원래 모델의 설정과 결과는 [모델 카드](../model-cards/)에 기록되어 있습니다.
