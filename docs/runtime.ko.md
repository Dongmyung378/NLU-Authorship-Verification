# CPU 실행과 저장 모델 재검증

[English](./runtime.md) | [프로젝트 개요](../README.ko.md) | [평가 규칙](./evaluation.ko.md) | [실측 결과](./results.ko.md)

Windows와 Linux 실행기는 Docker, 로컬 GPU나 API key 없이 CPU 추론과 저장 모델 평가를 지원합니다. 프로젝트 내부의 64비트 Python 3.12 환경과 [`requirements-runtime.txt`](../requirements-runtime.txt)를 사용합니다. 기본 모델은 작은 문체 기반 앙상블이며, Transformer는 독립적으로 맞춘 보정값을 적용해 모델 선택 집합에서 결정하고 고정 기록에 저장한 배포용 RoBERTa-Large 파이프라인을 사용합니다. 임계값 0.5의 원시 비교에서 고른 모델과 다를 수 있으며, 어느 모델도 최종 평가 점수로 선택하지 않습니다.

Large 실행 가능성과 처리량 검사는 완료했습니다. 유지한 학습 조건은 microbatch 2, 유효 배치 32, 대응 최대 6 epoch와 공통 patience 3이며, 실제 경과 9시간과 배정 worker 합산 18시간 상한을 적용합니다. 두 손실은 대응 학습 6 epoch를 완료했습니다. 최종 모델 파일과 [전체 평가 고정 기록](../reports/evaluation-freeze.json)의 검증을 마쳤고, 모델 세 개의 최종 4,708쌍 고정 평가도 완료했습니다. [실측 결과](./results.ko.md)에서 수치를 확인할 수 있습니다. 모델 파일 21개를 Kaggle 데이터셋 버전 1로 공개했으며, 변경하지 않은 공개 CLI에서 전체 파일의 비로그인 HTTP 200 다운로드와 바이트 수 및 SHA256 검증을 통과했습니다. [전달 검증 기록](../reports/model-download-verification.json)에 근거를 제공합니다. 새 Windows와 Linux CPU 환경에서 저장 모델 전체를 실행한 추론 6개를 완료했으며, 자원 비용 관찰을 완료해 [CPU 요약](../reports/cpu-reproduction.json)에 연결했습니다. 같은 물리적 컴퓨터의 새 CLI 전체 프로세스를 측정했으며, 자원 측정의 정상 종료로 엄격한 수치 실패를 바꾸지 않습니다. 아래 명령은 지원하는 릴리스 실행 방법이며, 환경 구성이나 합성 테스트의 성공만으로 두 운영체제에서 실제 최종 모델을 재검증했다고 주장하지 않습니다.

Windows와 Linux에서 모델 세 개를 각각 전체 4,708쌍으로 실행한 CPU 추론 6개를 모두 완료했습니다. ID 순서, 분류와 판정 여부, 개수, 혼동행렬, F1과 정확도 및 고정한 연결 정보가 최초 기준 결과와 일치했습니다. 원래 비교 대상인 주요 실수 지표는 `1e-6` 안에 있었으며 모든 진단값의 비트 단위 일치를 뜻하지 않습니다. 문체 모델의 CPU 간 검사는 통과했습니다. Large 두 모델은 변경하지 않은 `1e-5` 기준의 GPU와 CPU 확률 대조에 실패했습니다. Custom focal의 보정 확률 최대 차이는 Windows `1.80045428e-4`, Linux `1.09622536e-4`였으며 CE의 실패 기록도 보존합니다. 분류가 일치하더라도 엄격한 수치 재현은 실패 상태를 유지합니다. 공통 eager attention과 FP32 추론은 그대로이며 장치별 연산 차이일 가능성은 있지만 원인은 입증하지 못했습니다. CPU 자원 비용 관찰을 완료해 [CPU 요약](../reports/cpu-reproduction.json)에 연결했습니다. 같은 물리적 컴퓨터의 새 CLI 전체 프로세스를 측정했으며, 자원 측정의 정상 종료로 엄격한 수치 실패를 바꾸지 않습니다. 공개 모델 21개의 별도 비로그인 전달 검증은 [전달 기록](../reports/model-download-verification.json)에 있습니다.

CPU 비용은 같은 선택 집합의 첫 32쌍, CPU 스레드 1, 배치 1, 준비 예측 5회와 시간 측정 3회 반복으로 Windows를 순차 측정한 뒤 Linux WSL2를 측정했습니다. [Windows CE 원본 기록](../reports/cpu-replay/windows/costs/ce.json)과 [Linux CE 원본 기록](../reports/cpu-replay/linux/costs/ce.json)은 전처리, 고정한 AB와 BA 두 번의 추론 및 보정을 포함하며 모델 로딩은 별도입니다. 전체 프로세스의 상주 메모리 최대치는 Windows PeakWorkingSetSize와 Linux GNU time의 maxRSS를 사용합니다. Windows PeakPagefileUsage는 최대 프로세스 커밋 메모리이며 실제 디스크 스와핑 양을 측정한 값이 아닙니다. 무작위로 뽑지 않은 32쌍, 운영체제별 메모리 집계와 같은 물리적 컴퓨터라는 조건 때문에 성능 비교에는 한계가 있습니다. Windows와 Linux CPU 사이의 Large 두 모델 확률 비교도 변경하지 않은 허용치에서 실패했으며 모든 분류와 판정 여부는 일치했습니다. [CPU 요약](../reports/cpu-reproduction.json)에 최초 Kaggle 비교와 구분해 보존했습니다.

## 환경 구성과 다운로드

Kaggle 인증 정보 없이 공개 모델을 자동으로 다운로드할 수 있습니다. 명세는 데이터셋 버전 1을 고정하며 모든 파일은 고정한 크기와 SHA256이 일치해야 합니다. 일치하는 릴리스 소스와 명세에서 아래 명령을 실행합니다.

저장소 최상위 폴더에서 Windows PowerShell로 실행합니다.

```powershell
.\launch-nlu.bat doctor
.\launch-nlu.bat download
.\launch-nlu.bat download --verify-only
.\launch-nlu.bat verify-data
```

Linux에서는 다음 명령을 사용합니다.

```bash
sh ./launch-nlu.sh doctor
sh ./launch-nlu.sh download
sh ./launch-nlu.sh download --verify-only
sh ./launch-nlu.sh verify-data
```

최초 실행 시 적합한 Python 3.12가 없으면 관리형 Python을 내려받고, 고정한 패키지를 프로젝트 내부에 설치합니다. 시스템 전체의 Python 패키지는 바꾸지 않습니다. 기존 프로젝트 환경이 맞지 않으면 `.local/environment-backups/`에 보존한 후 교체합니다. 환경 구성에는 인터넷이 필요하며, 이후 저장 모델 추론은 로컬 토크나이저와 모델 파일을 사용합니다. 실행기는 로컬 GPU를 숨기고 실행 환경은 CPU를 명시적으로 사용합니다.

`doctor`는 고정 패키지 버전과 환경 격리, 작은 CPU tensor 연산을 확인합니다. 실제 가중치를 불러오거나 과제 정확도를 재현하지는 않습니다. `download`는 인증 정보를 포함하지 않는 HTTPS 파일 또는 ZIP의 바이트 수, SHA256과 전체 파일 목록을 확인한 뒤 설치합니다. 나눈 파일은 각 부분 해시와 다시 합친 전체 파일 해시도 검사합니다. `download --verify-only`는 다시 다운로드하지 않고 설치한 파일을 검사합니다. `verify-data`는 CSV 바이트, 원본 행, 집합 배정과 실제 전처리 문서의 분리를 검사합니다. 각 검사의 목적은 다릅니다.

릴리스 다운로드 명세 [`reports/artifact-download.json`](../reports/artifact-download.json)에 실제 공개 주소, 전체 모델 및 토크나이저 파일, 바이트 수와 SHA256을 연결합니다. 모델 파일만 포함한 공개 Kaggle 데이터셋의 버전 1 raw 파일을 비로그인으로 내려받으며 `kind=file`로 직접 연결합니다. 원격 파일 이름을 원래 실행 경로에 연결하므로 중첩 압축 파일을 풀어 모델 그룹을 구성하지 않습니다. 현재 주소와 해시는 실제 공개 후 전체 비로그인 HTTP 200 다운로드와 바이트 수 및 SHA256 검증을 통과했습니다. 비공개 학습 데이터셋은 공개 모델 주소가 아니며 명세나 주소가 없으면 해당 릴리스가 아직 제공되지 않는 상태입니다.

모델은 `artifacts/runtime/models/stylometric`, `ce`, `custom_asl` 아래에 설치하며, 그룹별로 내려받을 수 있습니다.

```powershell
.\launch-nlu.bat download --model transformer
.\launch-nlu.bat download --model ce
.\launch-nlu.bat download --model custom_asl
.\launch-nlu.bat download --model all
```

특정 그룹의 설치 파일을 검사하려면 같은 `--model` 값에 `--verify-only`를 추가합니다. `transformer`는 `reports/evaluation-freeze.json`에 기록한 모델 선택을 따릅니다. 기본 앙상블만 내려받으면 두 Large Transformer가 함께 설치되는 것은 아니므로 재검증할 모델의 그룹을 내려받아야 합니다. 대표 Transformer로 한 손실을 선택하더라도 두 FP32 체크포인트와 토크나이저 및 설정 파일을 모두 제공합니다. 실제 파일 크기는 명세를 따르며, Large 가중치 두 개는 부속 파일을 제외하고 합계 약 2.84 GB입니다.

### 이미 확보한 검증된 패키지의 수동 설치

모델 파일만 포함한 `nlu-frozen-stylometric.zip`, `nlu-frozen-ce.zip`, `nlu-frozen-custom_asl.zip` 세 개를 로컬에 준비했습니다. 이 수동 ZIP의 별도 공개 주소는 제공하지 않으며 공개 raw 파일은 자동 다운로드로 설치할 수 있습니다. 검증된 패키지를 이미 확보했다면 같은 릴리스의 프로젝트 최상위 폴더에 놓습니다. 각각 `models/<model-name>/` 아래에 고정 파일 7개를 포함하므로 실행기 설정 후 `artifacts/runtime`에 풉니다.

```powershell
.\launch-nlu.bat doctor
.\.venv\Scripts\python.exe -I -m zipfile -e nlu-frozen-stylometric.zip artifacts/runtime
.\.venv\Scripts\python.exe -I -m zipfile -e nlu-frozen-ce.zip artifacts/runtime
.\.venv\Scripts\python.exe -I -m zipfile -e nlu-frozen-custom_asl.zip artifacts/runtime
```

기본 사용에는 문체 패키지만 필요하며 Large 두 개는 선택적인 비교용입니다. Linux에서는 같은 압축 파일과 대상에 `.venv/bin/python -I -m zipfile -e`를 사용합니다. 수동 설치에는 `reports/artifact-download.json`이 필요하지 않으며, `predict`와 `evaluate`가 모델을 불러오기 전에 전체 파일을 고정 기록과 직접 대조합니다. 일치하는 개선 소스와 고정 기록이 필요하고 최초 Drive 모델은 다른 형식입니다. 고정 버전 명세를 사용하는 공개 자동 다운로드도 제공됩니다.

## 글 쌍 판별

```powershell
.\launch-nlu.bat predict
.\launch-nlu.bat predict --model transformer
.\launch-nlu.bat predict --text-file-1 passage-a.txt --text-file-2 passage-b.txt
.\launch-nlu.bat predict --model transformer --model-name ce --model-dir artifacts/runtime/models/ce --text-file-1 passage-a.txt --text-file-2 passage-b.txt
```

입력이 없는 글은 대화형으로 요청합니다. 자동 실행에서는 두 글 또는 UTF-8 파일 두 개를 함께 지정합니다. 입력 파일은 각각 2 MiB로 제한하며, 공통 전처리는 1,000,000자를 넘는 입력, NUL 문자, 전처리 후 빈 글도 거부합니다. 짧은 글이 프로그램에서 허용되더라도 분류 결과가 신뢰할 만하다는 뜻은 아닙니다. Transformer는 두 글의 합을 최대 512토큰으로 제한합니다.

JSON 출력은 숫자형 클래스 예측과 실제 판정인 `decision`을 구분합니다. 판정은 `same_author`, `different_authors`, `uncertain` 중 하나이며, `abstained=true`이면 판정을 보류합니다. `same_author_score`는 보정된 클래스 1 점수이고, `confidence_for_predicted_class`는 분류 임계값으로 선택한 클래스에 부여한 점수입니다. 고정한 임계값, temperature와 추론 방식도 표시합니다. 제공된 데이터에서 점수를 보정했다고 해서 새로운 저자의 신원을 증명하는 값이 되지는 않습니다.

저장 파일을 불러오기 전에 실행 환경은 모델 폴더와 실행 소스를 최종 평가 고정 기록과 대조합니다. 문체 모델의 pickle 파일을 해석하기 전에도 이 검사를 수행합니다. 보정값은 고정한 보고서에 두며 변경하지 않는 모델 파일 목록과 분리합니다. 설치 후에는 모델과 토크나이저를 로컬에서 불러옵니다. 최초 RoBERTa-Large와 문체 모델의 Drive 파일은 역사 노트북용으로 새로운 실행 환경의 고정 모델과 서로 바꿔 사용할 수 없습니다.

## 최종 평가 재검증

재검증마다 아직 없는 출력 폴더를 사용합니다.

```powershell
.\launch-nlu.bat evaluate --model stylometric --model-name stylometric --model-dir artifacts/runtime/models/stylometric --output .local/replay-stylometric --batch-size 4 --threads 4
.\launch-nlu.bat evaluate --model transformer --model-name ce --model-dir artifacts/runtime/models/ce --output .local/replay-ce --batch-size 4 --threads 4
.\launch-nlu.bat evaluate --model transformer --model-name custom_asl --model-dir artifacts/runtime/models/custom_asl --output .local/replay-custom-asl --batch-size 4 --threads 4
```

위 전체 재검증 예시는 배치 4와 CPU 스레드 4를 명시합니다. 새 모델의 성능 결과를 확인하기 전에 기존 모델의 작은 시간 측정으로 정한 실행 설정이며, 모델 선택이나 보정값을 바꾸지 않습니다. CLI 기본값은 배치 8과 스레드 1을 유지합니다. 최종 모델의 실제 CPU 실행 비용은 [CPU 요약](../reports/cpu-reproduction.json)과 [실측 결과](./results.ko.md)에 제시하며, 별도 속도 비교는 배치 1과 스레드 1을 사용했습니다.

기본 평가 집합은 `locked_holdout`이며 저장한 명세에서 문서를 분리한 글 쌍 4,708개를 포함합니다. 기본 분리 명세와 고정 기록은 `reports/split-manifest.json`과 `reports/evaluation-freeze.json`입니다. 소스, 데이터, 모델과 보정의 연결을 확인한 뒤 지표, 혼동행렬과 원본 순서의 예측을 저장합니다. 공개 가중치를 다시 불러오며 모델, temperature나 임계값을 다시 학습하지 않습니다. 고정 이후 실행 파일이나 고정 설정을 바꾸면 해당 릴리스의 재검증은 의도적으로 중단됩니다. 재현할 때는 공개한 소스 스냅샷을 그대로 사용합니다.

저장한 출력을 [보고서 비교 도우미](./reference-code/compare_frozen_predictions.py)로 대조합니다. 아래 명령은 실행기가 만든 프로젝트 내부 `.venv` 인터프리터를 사용하며 전역 `python` 명령에 의존하지 않습니다. 표준 라이브러리만 사용하므로 기존 Python 3.12 인터프리터로도 실행할 수 있습니다.

```powershell
.\.venv\Scripts\python.exe -I docs/reference-code/compare_frozen_predictions.py --model-name stylometric --evaluation-dir .local/replay-stylometric --output .local/compare-stylometric.json
.\.venv\Scripts\python.exe -I docs/reference-code/compare_frozen_predictions.py --model-name ce --evaluation-dir .local/replay-ce --output .local/compare-ce.json
.\.venv\Scripts\python.exe -I docs/reference-code/compare_frozen_predictions.py --model-name custom_asl --evaluation-dir .local/replay-custom-asl --output .local/compare-custom-asl.json
```

Linux에서는 같은 인자에 `.venv/bin/python -I`를 사용합니다. 전체 4,708개 ID 순서, 예측 파일 해시와 고정 소스, 분리 명세, 모델, 보정 및 클래스 연결을 확인합니다. 예측, 판정 여부, 그룹, 길이, 개수, F1과 정확도는 정확히 일치해야 하며 확률은 `1e-5`, 나머지 실수 지표는 `1e-6`의 고정 절대 오차를 적용합니다. 새 비교 JSON을 저장하고 통과 시 0, 실패 시 1로 종료합니다. 기준 평가는 문체 모델에는 Kaggle CPU, Large에는 CUDA/T4를 사용했으므로 JSON 바이트가 같아야 하는 검사는 아닙니다. 모델을 불러오거나 임계값을 조정하지 않으며 공개 다운로드도 검증하지 않습니다. 분류와 판정 여부가 모두 같아도 수치 허용치를 넘으면 비교 결과는 실패입니다. 실패 기록과 최초 기준 결과를 보존하며, 과제 점수의 일치로 `status=failed`를 통과로 바꾸지 않습니다.

전체 재검증은 실제 저장 모델을 모든 고정 기록에서 실행하고 지표 및 순서가 있는 출력을 릴리스 보고서와 비교해야 합니다. `doctor`, 다운로드 무결성 검사, 단일 예측이나 작은 모델 테스트는 이를 대신하지 않습니다. Large 체크포인트 두 개의 CPU 평가는 상당한 작업이며 시간과 메모리는 하드웨어에 따라 달라집니다. GPU 실행 가능성 검사에서 CPU 속도나 메모리 사용량을 보장하지 않습니다.

세 고정 예측 보고서가 모두 생긴 뒤 [`scripts/bootstrap_results.py`](../scripts/bootstrap_results.py)로 모델을 불러오지 않고 사전에 정한 불확실성 계산을 재현할 수 있습니다. 분리 명세, 고정 기록, 완료한 릴리스 요약과 세 모델의 원본 순서 예측 파일을 받아 연결을 확인하고, seed 378의 대응 문서 그룹 표집 1,000회를 수행합니다. 95% percentile 구간은 제공된 데이터와 학습 seed 한 번에 관한 것이며 새 저자 및 도메인이나 반복 학습 안정성을 검증하지 않습니다. 이 계산은 임계값을 맞추거나 배포 모델을 선택하지 않습니다. 정확히 일치하는 로컬 고정 파일로 새 Windows/Linux 환경에서 저장 모델 전체를 실행한 CPU 추론 6개를 완료했습니다. 두 환경은 같은 물리적 컴퓨터를 사용하므로 같은 호스트의 별도 환경이며, 독립된 제3자 컴퓨터 검증은 아닙니다.

## 테스트와 속도 측정

환경 구성 후 Windows에서 합성 데이터의 정확성 테스트를 실행합니다.

```powershell
.\.venv\Scripts\python.exe -I -m unittest discover -s tests -v
```

Linux에서는 `.venv/bin/python -I -m unittest discover -s tests -v`를 사용합니다. 구현 조건과 작은 저장 모델의 재로딩을 확인하는 테스트이며, 실제 데이터의 최종 정확도를 평가하는 것은 아닙니다.

```powershell
.\launch-nlu.bat benchmark --model stylometric --model-name stylometric --model-dir artifacts/runtime/models/stylometric --output .local/benchmark-stylometric.json
.\launch-nlu.bat benchmark --model transformer --model-name ce --model-dir artifacts/runtime/models/ce --output .local/benchmark-ce.json
```

기본 속도 측정은 정해진 모델 선택 사례, 배치 크기 1, 예열 5회, 반복 3회와 CPU 스레드 1개를 사용합니다. 정규화, 토큰화 또는 특징 추출, 고정한 추론 방식과 보정을 포함하며, 모델 로딩 시간은 별도로 기록합니다. 같은 하드웨어와 같은 사례에서 새 프로세스로 비교합니다. 대칭 추론은 두 번의 forward pass가 필요하므로 그 비용을 구분합니다. 특정 환경에서 검증됐다는 주장은 실제 환경 기록과 저장 모델 재검증 결과가 있어야 뒷받침됩니다.
