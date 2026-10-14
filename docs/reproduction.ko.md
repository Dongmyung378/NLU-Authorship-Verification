# 재현 가이드

[English](./reproduction.md) | [프로젝트 개요](../README.ko.md) | [실행 환경](./runtime.ko.md) | [평가 규칙](./evaluation.ko.md) | [실측 결과](./results.ko.md)

새로운 문서 분리 실험의 고정 파일을 다시 실행하는 가이드입니다. 기본 모델은 CPU 문체 앙상블이며 두 RoBERTa-Large 체크포인트는 선택적으로 내려받습니다. Windows와 Linux에서 격리된 고정 Python 3.12 CPU 환경을 사용하며 Docker, API key나 로컬 GPU 구성이 필요하지 않습니다. 재학습은 별도의 Kaggle 절차입니다.

**릴리스 상태:** 실행 가능성과 처리량 검사를 완료해 microbatch 2와 대응 최대 6 epoch를 유지했습니다. 같은 실행 및 설정 파일 35개와 격리된 Python 3.12.15 CPU 환경으로 Windows 11에서 109/109, Ubuntu 24.04 WSL2에서 109/109 구현 검사를 통과했습니다. 두 환경은 같은 물리적 컴퓨터를 사용하며 외부 제3자 검토를 뜻하지 않습니다. Large 두 손실은 대응 학습 6 epoch를 완료했으며, 모델 세 개의 최종 4,708쌍 고정 평가도 마쳤습니다. 완료한 실행의 수치는 [실측 결과](./results.ko.md)와 [실행 시간 기록](../reports/execution-timing.json)에 있습니다. 정확히 일치하는 로컬 고정 파일로 저장 모델 전체를 실행한 CPU 추론 6개를 완료했으며, 자원 비용 관찰을 완료해 [CPU 요약](../reports/cpu-reproduction.json)에 연결했습니다. 같은 물리적 컴퓨터의 새 CLI 전체 프로세스를 측정했으며, 자원 측정의 정상 종료로 엄격한 수치 실패를 바꾸지 않습니다. 모델 파일 21개를 Kaggle 데이터셋 버전 1로 공개했으며, 변경하지 않은 공개 CLI에서 전체 파일의 비로그인 HTTP 200 다운로드와 바이트 수 및 SHA256 검증을 통과했습니다. [전달 검증 기록](../reports/model-download-verification.json)에 근거를 제공합니다. 아래 명령은 확정한 실행 방법이며 합성 테스트 통과는 전체 저장 모델 평가와 다릅니다.

Windows와 Linux에서 모델 세 개를 각각 전체 4,708쌍으로 실행한 CPU 추론 6개를 모두 완료했습니다. ID 순서, 분류와 판정 여부, 개수, 혼동행렬, F1과 정확도 및 고정한 연결 정보가 최초 기준 결과와 일치했습니다. 원래 비교 대상인 주요 실수 지표는 `1e-6` 안에 있었으며 모든 진단값의 비트 단위 일치를 뜻하지 않습니다. 문체 모델의 CPU 간 검사는 통과했습니다. Large 두 모델은 변경하지 않은 `1e-5` 기준의 GPU와 CPU 확률 대조에 실패했습니다. Custom focal의 보정 확률 최대 차이는 Windows `1.80045428e-4`, Linux `1.09622536e-4`였으며 CE의 실패 기록도 보존합니다. 분류가 일치하더라도 엄격한 수치 재현은 실패 상태를 유지합니다. 공통 eager attention과 FP32 추론은 그대로이며 장치별 연산 차이일 가능성은 있지만 원인은 입증하지 못했습니다. CPU 자원 비용 관찰을 완료해 [CPU 요약](../reports/cpu-reproduction.json)에 연결했습니다. 같은 물리적 컴퓨터의 새 CLI 전체 프로세스를 측정했으며, 자원 측정의 정상 종료로 엄격한 수치 실패를 바꾸지 않습니다. 공개 모델 21개의 별도 비로그인 전달 검증은 [전달 기록](../reports/model-download-verification.json)에 있습니다.

CPU 비용은 같은 선택 집합의 첫 32쌍, CPU 스레드 1, 배치 1, 준비 예측 5회와 시간 측정 3회 반복으로 Windows를 순차 측정한 뒤 Linux WSL2를 측정했습니다. [Windows CE 원본 기록](../reports/cpu-replay/windows/costs/ce.json)과 [Linux CE 원본 기록](../reports/cpu-replay/linux/costs/ce.json)은 전처리, 고정한 AB와 BA 두 번의 추론 및 보정을 포함하며 모델 로딩은 별도입니다. 전체 프로세스의 상주 메모리 최대치는 Windows PeakWorkingSetSize와 Linux GNU time의 maxRSS를 사용합니다. Windows PeakPagefileUsage는 최대 프로세스 커밋 메모리이며 실제 디스크 스와핑 양을 측정한 값이 아닙니다. 무작위로 뽑지 않은 32쌍, 운영체제별 메모리 집계와 같은 물리적 컴퓨터라는 조건 때문에 성능 비교에는 한계가 있습니다. Windows와 Linux CPU 사이의 Large 두 모델 확률 비교도 변경하지 않은 허용치에서 실패했으며 모든 분류와 판정 여부는 일치했습니다. [CPU 요약](../reports/cpu-reproduction.json)에 최초 Kaggle 비교와 구분해 보존했습니다.

## 1. 공개한 소스 보존

정확한 릴리스 소스와 `reports/split-manifest.json`을 사용합니다. 평가 고정 기록이 실행 파일, 고정 요구사항과 실험 계획을 연결하므로 해당 파일을 변경하면 릴리스 재검증이 중단됩니다. 기록한 결과를 재현할 때 데이터 분리를 다시 만들거나 새로운 임계값을 맞추지 않습니다.

라벨이 있는 `data/train.csv`와 `data/dev.csv`는 새 분리 명세의 원본 입력이며 현재의 학습 및 모델 선택 구간 자체는 아닙니다. 명세는 31,388쌍을 유지해 학습 21,972쌍, 모델 선택 3,139쌍, 보정 1,569쌍과 최종 평가 4,708쌍을 배정합니다. 선언한 모든 전처리 방식에서 문서 동일성으로 연결한 그룹을 분리합니다. 저자 ID와 도메인 라벨이 없으므로 새 저자나 도메인의 일반화까지 확인한 것은 아닙니다.

## 2. 새 CPU 환경 구성

저장소 최상위 폴더에서 Windows PowerShell로 실행합니다.

```powershell
.\launch-nlu.bat doctor
.\launch-nlu.bat verify-data
```

Linux에서는 다음과 같이 실행합니다.

```bash
sh ./launch-nlu.sh doctor
sh ./launch-nlu.sh verify-data
```

실행기는 적합한 64비트 Python 3.12를 찾고 필요하면 관리형 실행기를 만든 뒤 로컬 `.venv`에 `requirements-runtime.txt`를 설치합니다. 시스템 패키지를 사용하거나 전역 Python을 변경하지 않습니다. 기존 환경이 맞지 않으면 `.local/environment-backups/`에 보존한 뒤 교체합니다. 최초 구성에는 인터넷이 필요하며 로컬 GPU를 숨기고 고정 CPU PyTorch를 사용합니다.

`doctor`는 패키지 버전, 환경 격리와 실제 작은 CPU tensor 연산을 확인하며 모델 파일이나 정확도를 검사하지 않습니다. `verify-data`는 CSV 해시, 원본 행, 집합 역할과 실제 전처리 문서의 분리를 확인합니다. 재현 결과를 기록할 때 실제 환경 보고서를 함께 보존합니다.

## 3. 선택한 모델 설치

### 이미 확보한 검증된 패키지의 수동 설치

모델 파일만 포함한 `nlu-frozen-stylometric.zip`, `nlu-frozen-ce.zip`, `nlu-frozen-custom_asl.zip` 세 개를 로컬에 준비했습니다. 이 수동 ZIP의 별도 공개 주소는 제공하지 않으며 공개 raw 파일은 자동 다운로드로 설치할 수 있습니다. 검증된 패키지를 이미 확보했다면 같은 릴리스의 프로젝트 최상위 폴더에 놓습니다. 각각 `models/<model-name>/` 아래에 고정 파일 7개를 포함하므로 실행기 설정 후 `artifacts/runtime`에 풉니다.

```powershell
.\launch-nlu.bat doctor
.\.venv\Scripts\python.exe -I -m zipfile -e nlu-frozen-stylometric.zip artifacts/runtime
.\.venv\Scripts\python.exe -I -m zipfile -e nlu-frozen-ce.zip artifacts/runtime
.\.venv\Scripts\python.exe -I -m zipfile -e nlu-frozen-custom_asl.zip artifacts/runtime
```

기본 사용에는 문체 패키지만 필요하며 Large 두 개는 선택적인 비교용입니다. Linux에서는 같은 압축 파일과 대상에 `.venv/bin/python -I -m zipfile -e`를 사용합니다. 수동 설치에는 `reports/artifact-download.json`이 필요하지 않으며, `predict`와 `evaluate`가 모델을 불러오기 전에 전체 파일을 고정 기록과 직접 대조합니다. 일치하는 개선 소스와 고정 기록이 필요하고 최초 Drive 모델은 다른 형식입니다. 고정 버전 명세를 사용하는 공개 자동 다운로드도 제공됩니다.



이 릴리스에 포함한 검증된 공개 다운로드 명세로 실행합니다.

```powershell
.\launch-nlu.bat download
.\launch-nlu.bat download --verify-only
```

기본 모델은 `artifacts/runtime/models/stylometric/`에 설치합니다. `reports/artifact-download.json`은 모델 파일만 포함한 공개 Kaggle 데이터셋의 비로그인 고정 버전 raw 주소를 `kind=file`로 기록합니다. `ce--model.safetensors` 같은 원격 파일 이름은 `models/ce/model.safetensors` 같은 원래 실행 경로에 연결합니다. 파일 21개 모두 전체 비로그인 HTTP 200 다운로드와 바이트 수 및 SHA256 검사를 통과했으며, [전달 검증 기록](../reports/model-download-verification.json)에 실제 CLI 근거를 보존했습니다.

자동 다운로드 및 검증 CLI는 기존 버전 1 raw 파일의 주소와 SHA256을 계속 고정합니다. [전체 ZIP 다운로드](https://www.kaggle.com/api/v1/datasets/download/dongmyungpark/nlu-authorship-frozen-large-20261012?datasetVersionNumber=2)는 동일한 고정 모델 파일 21개를 다시 묶은 데이터셋 버전 2에서 제공합니다. 다운로드한 ZIP의 파일 크기와 SHA256은 별도 [압축 파일 검증 기록](../reports/model-archive-verification.json)에서 확인할 수 있습니다. 모델과 평가 결과는 변경하지 않았습니다. 전체 ZIP은 Kaggle의 평면 파일 이름을 유지합니다. 원래 모델 실행 경로에 설치할 때는 이 안내의 자동 다운로드 명령을 사용합니다.

모든 모델, 변환기, 토크나이저, 설정과 파일 목록은 정확한 바이트 수와 SHA256에 일치해야 합니다. 다운로드는 검증한 파일을 임시 영역에 준비한 뒤 설치하고 기존 출력 폴더를 보존하며 위험한 경로나 심볼릭 링크를 거부합니다. 추론 전 모델 자체 목록과 신뢰한 평가 고정 기록을 다시 확인하며 문체 모델의 pickle을 해석하기 전에도 검사합니다.

선택적으로 다음 그룹을 설치합니다.

```powershell
.\launch-nlu.bat download --model transformer
.\launch-nlu.bat download --model ce
.\launch-nlu.bat download --model custom_asl
.\launch-nlu.bat download --model all
.\launch-nlu.bat download --model all --verify-only
```

`transformer`는 모델 선택 집합에서 결정하고 고정한 보정 파이프라인을 따릅니다. 비교 재검증을 위해 Large FP32 체크포인트 두 개를 모두 제공하며 가중치 합계는 부속 파일을 제외하고 약 2.84 GB입니다. 실제 크기는 명세에서 확인합니다. 기본 앙상블 다운로드에는 Transformer가 포함되지 않습니다.

## 4. 글 두 개 예측

```powershell
.\launch-nlu.bat predict
.\launch-nlu.bat predict --text-file-1 passage-a.txt --text-file-2 passage-b.txt
.\launch-nlu.bat predict --model transformer --text-file-1 passage-a.txt --text-file-2 passage-b.txt
```

Linux에서는 같은 옵션 앞에 `sh ./launch-nlu.sh`를 사용합니다. 자동 실행에서는 글 두 개 또는 UTF-8 파일 두 개를 함께 지정합니다. 입력은 글마다 2 MiB와 전처리 기준 1,000,000자로 제한하고 NUL 문자나 전처리 후 빈 글을 거부합니다. RoBERTa는 longest-first 방식으로 두 글 전체를 512토큰까지 제한합니다.

출력은 숫자형 클래스, 보정된 동일 저자 점수, 고정 temperature, 임계값과 입력 순서 정책을 표시합니다. `decision=uncertain`이면 고정한 보류 규칙으로 판정하지 않은 상태입니다. 제공된 데이터에서 보정한 점수는 새로운 글의 저자 신원을 입증하지 않습니다. 설치 후 모든 모델과 토크나이저 파일은 로컬에서 불러옵니다.

## 5. 고정한 최종 기록 전체 평가

모델마다 아직 없는 출력 폴더를 사용합니다.

```powershell
.\launch-nlu.bat evaluate --model stylometric --model-name stylometric --model-dir artifacts/runtime/models/stylometric --output .local/replay-stylometric --batch-size 4 --threads 4
.\launch-nlu.bat evaluate --model transformer --model-name ce --model-dir artifacts/runtime/models/ce --output .local/replay-ce --batch-size 4 --threads 4
.\launch-nlu.bat evaluate --model transformer --model-name custom_asl --model-dir artifacts/runtime/models/custom_asl --output .local/replay-custom-asl --batch-size 4 --threads 4
```

위 전체 재검증 예시는 배치 4와 CPU 스레드 4를 명시합니다. 새 모델의 성능 결과를 확인하기 전에 기존 모델의 작은 시간 측정으로 정한 실행 설정이며, 모델 선택이나 보정값을 바꾸지 않습니다. CLI 기본값은 배치 8과 스레드 1을 유지합니다. 최종 모델의 실제 CPU 실행 비용은 [CPU 요약](../reports/cpu-reproduction.json)과 [실측 결과](./results.ko.md)에 제시하며, 별도 속도 비교는 배치 1과 스레드 1을 사용했습니다.

기본 평가는 `locked_holdout` 4,708쌍 전체입니다. `reports/split-manifest.json`과 `reports/evaluation-freeze.json`을 기본으로 확인합니다. 고정 가중치와 보정을 다시 불러오되 학습하거나 판정 기준을 맞추지 않으며 `metrics.json`, `confusion.json`과 원래 순서의 `predictions.jsonl`을 저장합니다. 릴리스 보고서와 사례 수, 정답 수, 혼동행렬, 보정 지표, coverage와 원본 순서의 예측을 대조합니다. 모델, 소스, 데이터와 고정 기록 해시에 실제 OS/Python/패키지 버전도 함께 보존합니다.

평가 후 [보고서 비교 도우미](./reference-code/compare_frozen_predictions.py)로 대조를 자동화합니다. 아래 명령은 실행기가 만든 프로젝트 내부 `.venv` 인터프리터를 사용하므로 전역 `python` 명령이 필요하지 않습니다. 표준 라이브러리만 사용하는 검사이므로 기존 Python 3.12 인터프리터로도 실행할 수 있습니다. 비교 JSON은 실행마다 아직 없는 새 경로를 사용합니다.

```powershell
.\.venv\Scripts\python.exe -I docs/reference-code/compare_frozen_predictions.py --model-name stylometric --evaluation-dir .local/replay-stylometric --output .local/compare-stylometric.json
.\.venv\Scripts\python.exe -I docs/reference-code/compare_frozen_predictions.py --model-name ce --evaluation-dir .local/replay-ce --output .local/compare-ce.json
.\.venv\Scripts\python.exe -I docs/reference-code/compare_frozen_predictions.py --model-name custom_asl --evaluation-dir .local/replay-custom-asl --output .local/compare-custom-asl.json
```

Linux에서는 같은 도우미와 인자에 `.venv/bin/python -I`를 사용합니다. 전체 ID 순서, 예측 파일 SHA256과 분리 명세, 고정 기록, 실행 소스, 모델 파일 명세, 보정 및 클래스 연결을 확인합니다. 라벨, 예측, 판정 여부, 그룹, 길이, 개수, F1과 정확도는 정확히 일치해야 합니다. 확률의 허용 차이는 `1e-5`, 나머지 실수 지표는 `1e-6`이며 인자로 바꿀 수 없습니다. 종료 코드 0은 통과, 1은 실패이며 근거를 JSON으로 저장합니다. 기존 비교 파일을 덮어쓰거나 판정 기준을 다시 맞추지 않습니다. Kaggle 기준 평가는 문체 앙상블에는 CPU, Large에는 CUDA/T4를 사용했으므로 장치 간 JSON 바이트가 같아야 하는 검사는 아닙니다. 보고서만 대조하며 추론을 수행하거나 공개 다운로드 가능 여부를 입증하지 않습니다. 분류와 판정 사례가 일치해도 수치 검사 실패는 그대로 기록하며, 허용치와 최초 기준 결과를 바꾸지 않고 실패 근거를 보존합니다.

임계값 0.5의 원시 AB 비교와 배포용 보정 파이프라인의 선택 결과는 별도 보고서입니다. 둘 다 최종 평가 전에 고정하며 다른 Large 체크포인트를 평가했다고 해서 그 최종 점수로 대표 모델을 다시 고르지 않습니다.

새 Windows와 Linux 환경에서 정확히 일치하는 로컬 고정 파일로 저장 모델 전체를 실행한 CPU 추론 6개를 완료했습니다. 로컬 파일 재검증은 비로그인 공개 다운로드가 가능하다는 근거와 구분합니다. 두 환경이 같은 물리적 컴퓨터를 사용하면 같은 호스트의 별도 새 환경으로 기록하며 독립된 제3자 컴퓨터라고 표현하지 않습니다. Large CPU 평가의 시간과 메모리는 하드웨어에 따라 다르며 T4 검사는 로컬 CPU 추론 측정이 아닙니다.

## 6. 정확성, 시간과 불확실성 확인

환경 구성 후 합성 테스트를 실행합니다.

```powershell
.\.venv\Scripts\python.exe -I -m unittest discover -s tests -v
```

Linux에서는 `.venv/bin/python -I -m unittest discover -s tests -v`를 사용합니다. 그룹 분리, 손실과 보정, 파일 무결성과 작은 모델 재로딩을 확인하며 앞의 실제 모델 평가와 구분합니다.

```powershell
.\launch-nlu.bat benchmark --model stylometric --model-name stylometric --model-dir artifacts/runtime/models/stylometric --output .local/benchmark-stylometric.json
.\launch-nlu.bat benchmark --model transformer --model-name ce --model-dir artifacts/runtime/models/ce --output .local/benchmark-ce.json
```

기본 측정은 고정된 모델 선택 사례 32개, CPU 스레드 1개, 배치 1, 예열 5회와 반복 3회를 사용합니다. 전처리, 토큰화 또는 특징, 선택한 입력 순서와 보정을 포함하며 모델 로딩은 따로 기록합니다. 대칭 Large 추론은 forward pass를 두 번 실행합니다. 같은 하드웨어와 예시에서 새 프로세스로 비교합니다.

사전에 선언한 [bootstrap 도우미](../scripts/bootstrap_results.py)는 `--manifest`, `--freeze`, `--summary`, `--prediction-root`와 새로운 `--output` 경로를 받습니다. 완료한 릴리스 요약과 세 최종 예측 보고서가 있는 폴더를 지정합니다. 정확한 ID와 해시를 확인한 뒤 seed 378의 대응 문서 연결 요소 표집 1,000회로 조건부 95% percentile 구간을 계산합니다. 모델을 불러오거나 판정 기준을 다시 맞추고 선택하지 않습니다. 제공된 데이터와 seed 한 번에 대한 구간이며 새 저자, 도메인이나 반복 학습의 변동을 뜻하지 않습니다.

그림 생성은 선택 사항이며 별도의 Matplotlib 환경을 사용합니다. 모델 실행기는 이 패키지를 설치하지 않습니다. 완료한 릴리스 보고서가 있을 때 Windows에서 다음과 같이 실행할 수 있습니다.

```powershell
.\.venv\Scripts\python.exe -m venv .local/plotting-env
.\.local\plotting-env\Scripts\python.exe -m pip install -r docs/reference-code/requirements-plotting.txt
.\.local\plotting-env\Scripts\python.exe docs/reference-code/plot_frozen_results.py --summary reports/release-summary.json --output .local/frozen-evaluation.png
```

Linux에서는 `.venv/bin/python`으로 환경을 만들고 나머지 명령에 `.local/plotting-env/bin/python`을 사용합니다. 저장된 보고서만 읽어 그림을 생성하며 모델을 불러오거나 보정값과 CPU 패키지를 바꾸지 않습니다. 릴리스와 모델 평가가 먼저 완료돼야 하며 그림 생성으로 실제 평가를 대신하지 않습니다.

새 대응 학습은 [Kaggle 학습 가이드](./training.ko.md)를 사용합니다. RoBERTa-Large를 유지하고 제한된 문체 실험은 CPU에서 따로 수행한 뒤, 검증한 모델을 두 T4 손실 worker의 입력으로 패키징합니다.

## 최초 노트북과 Drive 파일

2026년 2월부터 5월까지의 공동 개발 기록은 `notebooks/`, [포스터](./authorship-verification-poster-preview.png)와 [최초 모델 카드](../model-cards/)에 보존합니다. 당시 개발 절차를 설명하는 자료이며 새로운 실행 릴리스나 평가 규칙이 아닙니다. 최초 노트북은 변경하지 않았으며 당시 노트북 의존성과 데모로 새로운 고정 CPU 환경을 덮어쓰지 않습니다.

| 최초 파일 | 원래 다운로드 | 당시 로컬 폴더 |
|---|---|---|
| 문체 앙상블 | [최초 모델과 변환기](https://drive.google.com/drive/folders/1mF0bMpzfs3XODpQA2Hd7H_JvXLk5AGgp?usp=sharing) | `models/stylometric-ensemble/` 또는 `models/Models_ML/` |
| RoBERTa-Large 직접 구현 비대칭 focal | [최초 체크포인트와 토크나이저](https://drive.google.com/drive/folders/1Ty9-EpOLAgzvTWPuqrHBTbehmZZuDs3V?usp=sharing) | `models/roberta-asymmetric-loss/` 또는 `models/Models_ASL/` |

최초 개발 5,993쌍에는 학습 글 쌍과 완전히 같은 630쌍이 있습니다. 이 완전 중복을 포함해 총 2,738쌍이 학습 문서를 하나 이상 재사용하며, 한 문서 재사용은 1,658쌍, 두 문서 재사용은 1,080쌍입니다. 최초 문체 노트북은 학습과 개발 글을 합쳐 TF-IDF를 맞추므로 어휘와 IDF에 개발 글이 반영됩니다. 체크포인트, 가중치와 임계값도 개발 집합에서 선택합니다. 당시 점수는 개발 결과이며 새 최종 평가 성능의 독립적인 비교 기준이 아닙니다.

최초 직접 구현 이진 focal은 ASL 또는 ASLoss라고 불렸습니다. Gamma 2/1의 두 클래스 softmax focal 가중치를 사용하는 ASL의 이진 변형을 직접 구현했으며, 공식 기본 설정이나 논문의 실험을 그대로 재현했다고 주장하지 않습니다. [명칭 안내](./loss-terminology.ko.md)에서 수치 안정성 처리의 차이와 호환 식별자를 설명합니다. 저장한 테스트 예측 5,985개와 추가 live 20쌍은 정답이 없어 정확도나 F1을 제공하지 않습니다. 문서 분리, 학습 집합만의 TF-IDF와 고정 평가는 새로운 규칙의 타당성을 개선하지만 과거 데이터 사용을 없애거나 새 저자 및 도메인의 일반화를 입증하지는 않습니다.
