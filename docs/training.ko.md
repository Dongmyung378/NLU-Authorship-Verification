# Kaggle에서 조건을 통제한 학습

[English](./training.md) | [프로젝트 개요](../README.ko.md) | [평가 규칙](./evaluation.ko.md) | [실측 결과](./results.ko.md)

공개 CPU 실행 환경은 저장된 모델을 다시 실행합니다. Transformer 과제 학습은 Kaggle GPU에서 수행하고, 문체 기반 실험은 격리된 로컬 CPU 환경에서 별도로 진행합니다. 2026년 2월부터 5월까지 Dongmyung Park과 Juho Kim이 공동 개발한 최초 RoBERTa-Large와 문체 모델 노트북은 역사 자료로 보존합니다. 새 규칙도 두 모델 계열을 유지하며 문서를 분리한 명세에서 다시 학습합니다. 완료한 Large 실행 가능성 검사는 [`reports/capability/`](../reports/capability/)에 있습니다. 두 손실은 대응 학습 6 epoch를 완료하고 각각 5 epoch 체크포인트를 저장했으며, 모델 세 개의 고정 최종 평가도 완료했습니다. [실측 결과](./results.ko.md)와 [실행 시간 기록](../reports/execution-timing.json)에 근거를 제공합니다. 모델 파일 21개를 Kaggle 데이터셋 버전 1로 공개했으며, 변경하지 않은 공개 CLI에서 전체 파일의 비로그인 HTTP 200 다운로드와 바이트 수 및 SHA256 검증을 통과했습니다. [전달 검증 기록](../reports/model-download-verification.json)에 근거를 제공합니다. 새 Windows와 Linux 환경에서 저장 모델 전체를 실행한 CPU 추론 6개를 완료했으며, 자원 비용 관찰을 완료해 [CPU 요약](../reports/cpu-reproduction.json)에 연결했습니다. 같은 물리적 컴퓨터의 새 CLI 전체 프로세스를 측정했으며, 자원 측정의 정상 종료로 엄격한 수치 실패를 바꾸지 않습니다.

Windows와 Linux에서 모델 세 개를 각각 전체 4,708쌍으로 실행한 CPU 추론 6개를 모두 완료했습니다. ID 순서, 분류와 판정 여부, 개수, 혼동행렬, F1과 정확도 및 고정한 연결 정보가 최초 기준 결과와 일치했습니다. 원래 비교 대상인 주요 실수 지표는 `1e-6` 안에 있었으며 모든 진단값의 비트 단위 일치를 뜻하지 않습니다. 문체 모델의 CPU 간 검사는 통과했습니다. Large 두 모델은 변경하지 않은 `1e-5` 기준의 GPU와 CPU 확률 대조에 실패했습니다. Custom focal의 보정 확률 최대 차이는 Windows `1.80045428e-4`, Linux `1.09622536e-4`였으며 CE의 실패 기록도 보존합니다. 분류가 일치하더라도 엄격한 수치 재현은 실패 상태를 유지합니다. 공통 eager attention과 FP32 추론은 그대로이며 장치별 연산 차이일 가능성은 있지만 원인은 입증하지 못했습니다. CPU 자원 비용 관찰을 완료해 [CPU 요약](../reports/cpu-reproduction.json)에 연결했습니다. 같은 물리적 컴퓨터의 새 CLI 전체 프로세스를 측정했으며, 자원 측정의 정상 종료로 엄격한 수치 실패를 바꾸지 않습니다.

## 고정한 학습 조건

[`configs/experiment-plan.json`](../configs/experiment-plan.json)은 전체 과제 학습 전에 실험을 선언합니다. `FacebookAI/roberta-large`의 리비전 `722cf37b1afa9454edce342e7895e588b6ff1d59`와 사전학습 safetensors SHA256 `047c85f0b96269cd62e6f732644f067004eebd95af5b5d35965ae2528f13bf38`을 고정합니다. 이진 분류 모델의 파라미터는 355,361,794개입니다. 두 손실 모두 최초 과제 체크포인트가 아닌 동일한 새 과제 초기화에서 시작합니다.

고정한 공통 학습 조건은 seed 378, 전체 길이 512, microbatch 2, 유효 배치 32, 대응 최대 6 epoch, learning rate `5e-6`, weight decay `0.01`, warmup ratio `0.1`, gradient norm 상한 `1.0`과 gradient checkpointing입니다. 완료한 처리량 검사에서는 더 큰 microbatch를 채택할 근거가 부족했습니다. 최대 6 epoch를 유지해 환경 구성, 모델 선택, 체크포인트, 보정, 고정 평가와 실패 대비를 위한 보수적 예산을 확보합니다. 검증 점수 없이 상한을 정하며 두 손실의 학습 전에 고정합니다. CE와 custom ASL은 같은 최초 가중치 해시와 epoch별 순서를 사용합니다. 학습 목적에서 의도한 차이는 최초의 클래스별 focal 가중치로, 클래스 0에는 gamma 2, 클래스 1에는 gamma 1을 적용합니다.

각 epoch는 고정된 난수 순서로 섞고, 사례를 두 배로 늘리지 않으면서 약 절반의 글 쌍에서 입력 순서를 바꿉니다. 교환 여부는 `seed:epoch:pair_id`의 SHA256 첫 바이트에서 가장 낮은 비트로 정합니다. Gradient accumulation은 마지막의 작은 유효 배치까지 실제 사례 수에 맞춰 microbatch를 가중합니다. Forward pass는 혼합 정밀도를 사용하고 손실은 FP32로 계산합니다. Overflow로 건너뛴 업데이트를 성공한 optimizer step으로 계산하지 않고 기록합니다.

## 실제 확인한 Large 실행 가능성

별도의 비공개 Kaggle 검사는 Python 3.12.15, PyTorch 2.5.1+cu124와 CUDA 12.4 환경의 T4에서 완료했습니다. 유효 배치 update 40회를 시도해 39회 성공했고 FP16 scaler가 1회를 건너뛰었습니다. 시도 시간 중앙값은 5.136초, 최대 할당 메모리는 6.09 GiB, 최대 예약 메모리는 6.37 GiB였습니다. OOM 재시도는 없었습니다. 검사 가중치는 모두 폐기했으며 모델 선택, 보정이나 최종 추론을 수행하지 않았습니다.

완료한 검사는 당시의 3-epoch 직렬 실행을 추정했습니다. 30% 여유를 포함한 7.65시간은 그 조건의 추정값으로 보존하며, 이후 전체 규칙의 실제 실행 시간으로 표현하지 않습니다. 전체 학습 시간은 실제 결과로 확인하며 검사 수치로 보장하지 않습니다.

## 제한된 처리량 검사

과제 학습 전에 별도의 시간 측정 검사로 microbatch 후보 2, 8, 32를 비교하고 유효 배치 32는 유지했습니다. 각 후보는 같은 seed의 학습 기록 순서에서 optimizer update 20회를 시도했습니다. 사전 규칙은 후보당 절반 크기의 OOM 재시도 1회, 채택에 필요한 처리량 배수 1.05 이상과 GPU 용량의 90% 이하 예약 메모리를 선언했습니다. 모든 후보는 같은 고정 사전학습 가중치와 새 초기화에서 시작하고 바뀐 가중치를 모두 폐기했습니다. 검증 추론, 모델 성능 비교, 보정이나 최종 평가 추론은 수행하지 않았습니다.

| Microbatch | 시도 시간 중앙값 | 최대 할당 메모리 | 성공 update | Overflow 생략 | OOM 재시도 |
|---|---:|---:|---:|---:|---:|
| 2 | 4.9123초 | 6.09 GiB | 19 / 20 | 1 | 0 |
| 8 | 4.7953초 | 6.66 GiB | 19 / 20 | 1 | 0 |
| 32 | 5.4535초 | 7.94 GiB | 19 / 20 | 1 | 0 |

검사는 316.29초에 완료했습니다. Microbatch 8의 처리량 배수 1.0244는 채택 기준 1.05에 미치지 못했고 32는 더 느렸으므로 microbatch 2를 유지했습니다. 측정 바이트는 [`throughput-profile.json`](../reports/capability/throughput-profile.json)과 [`요약`](../reports/capability/throughput-summary.json)에 보존합니다. 학습 기록만 사용한 자원 측정은 정확도를 제공하거나 손실을 선택하지 않습니다. 정제 데이터 명세와 소스 스냅샷을 연결하며, 완료한 v3 실행 가능성 증거는 변경하지 않고 보존합니다.

## 대응 병렬 학습과 예산

부모 프로세스는 자신의 Kaggle kernel에 이미 배정된 두 T4에서 독립된 손실 worker 두 개를 시작합니다. 각 자식 프로세스는 자신에게 배정된 GPU만 볼 수 있습니다. 하나의 모델을 data parallel로 학습하는 방식이 아니라 CE와 ASL 두 모델을 각각 병렬 학습합니다. 다른 kernel이나 GPU 배정은 바꾸지 않습니다.

Worker는 매 epoch가 완료될 때 서로의 결과를 기다립니다. 임계값 0.5에서 원래 입력 순서인 AB의 모델 선택 macro F1이 둘 다 3 epoch 연속 개선되지 않으면 공통 patience 조건으로 종료합니다. 어느 하나라도 개선되면 공통 대기 횟수를 초기화합니다. 이 규칙이나 고정한 최대 6 epoch에서 두 모델이 함께 멈추므로 완료한 epoch 수를 같게 유지합니다. 각 손실은 자신의 최적 체크포인트를 다시 불러와 AB와 BA 모델 선택 logits를 생성하고, 정확한 동점에서는 더 이른 epoch를 유지합니다. 학습 함수는 보정을 맞추거나 최종 평가 집합에서 추론하지 않습니다.

대응 학습은 실제 경과 최대 9시간과 두 배정 worker 시간 합산 최대 18시간으로 제한합니다. Worker 합산 시간은 GPU가 배정된 동안의 준비와 서로 기다린 시간도 포함합니다. 환경 구성과 릴리스 역시 세션 시간을 사용하므로 부모 프로세스의 학습 제한만으로 전체 세션 예산을 계산하지 않습니다. 자체적인 보수적 예산이며 Kaggle의 공식 T4x2 사용량 배수를 검증했다는 의미는 아닙니다. 부모 프로세스가 두 학습 제한을 모두 적용합니다. 한 worker가 실패하거나 완료하지 못하면 대응 실험 전체를 실패로 처리하고 부분 결과는 보존하되 성공한 비교나 릴리스로 공개하지 않습니다.

실제 전체 커널에도 36,000초의 세션 제한을 적용합니다. 환경 구성부터 대응 학습과 릴리스까지 실제 경과 10시간, 배정된 GPU 두 개의 시간을 합산하면 최대 20시간으로 제한합니다. 앞서 완료한 검사에는 별도로 최대 2시간의 합산 여유를 두어 총 22시간을 넘기지 않도록 합니다. 릴리스가 중단되면 완료한 평가로 표시하지 않습니다.

## CPU 문체 기반 실험

`nlu_authorship.stylometric.STUDY_PLAN`은 설정 4개, 최대 XGBoost/LightGBM 구성 모델 학습 8회, seed 378, CPU 스레드 2개와 전체 120분을 고정합니다. 단어, 문자와 기능어 TF-IDF는 학습 집합에서만 맞춥니다. 세 pilot은 전체 특징, TF-IDF 특징만, 문체 특징만 사용하는 경우를 tree 상한 500으로 비교합니다. 전체 특징 설정 하나에는 최초의 더 큰 상한인 XGBoost 7,000개와 LightGBM 2,000개 tree를 유지합니다. 모델 선택 logloss의 early stopping과 학습 시간 callback으로 실행을 제한합니다.

전체 특징 50/50 pilot이 기준입니다. 새로운 설정이나 가중치는 임계값 0.5의 모델 선택 macro F1이 0.002 이상 증가해야 이를 대체합니다. 최종 XGBoost 가중치는 0.25, 0.5, 0.75 중 하나이며 0과 1은 단일 구성 모델의 진단으로만 사용합니다. 보정과 최종 평가 집합은 사용하지 않습니다. 선택한 앙상블은 모든 TF-IDF 변환기, 설정과 파일 해시를 함께 저장하고 다시 불러와 선택 점수 및 실제 입력 교환의 불변성을 확인합니다.

CPU 환경을 구성한 뒤 프로젝트 루트에서 새로운 실험을 실행합니다. 실험 전에 CPU 스레드 2개를 초기화합니다. 최소 구성의 Linux에서도 LightGBM보다 먼저 고정된 Torch 패키지에 포함된 OpenMP를 불러오므로 시스템 패키지를 전역으로 설치할 필요가 없습니다.

Windows:

```powershell
.\.venv\Scripts\python.exe -I -c "from nlu_authorship.runtime import configure_cpu; configure_cpu(2); from nlu_authorship.stylometric import train_stylometric; train_stylometric('.', 'reports/split-manifest.json', '.local/stylometric')"
```

Linux:

```bash
.venv/bin/python -I -c "from nlu_authorship.runtime import configure_cpu; configure_cpu(2); from nlu_authorship.stylometric import train_stylometric; train_stylometric('.', 'reports/split-manifest.json', '.local/stylometric')"
```

모델 출력과 같은 상위 폴더의 실험 기록 폴더는 아직 없어야 합니다. 저장한 CPU 모델을 전체 Kaggle 패키지에 제공하므로 GPU 작업에서 이 실험을 다시 학습하지 않습니다.

## 실행 패키지 준비

도우미는 정확한 바이트 목록과 비공개 Kaggle 메타데이터를 생성하며 GPU 작업을 직접 시작하지는 않습니다. 자신의 계정에 속한 ID와 아직 없는 출력 폴더를 사용합니다.

```powershell
.\.venv\Scripts\python.exe -I -m scripts.package_kaggle --output .local/kaggle-full-package --dataset-id YOURNAME/nlu-large-source --kernel-id YOURNAME/nlu-large-paired --stylometric-model .local/stylometric
```

생성한 `dataset/`을 비공개 데이터셋으로 올리고 `kernel/` 스크립트를 인터넷과 T4 두 개가 배정된 비공개 GPU kernel로 실행합니다. 자신의 Kaggle 도구로 인증하고 인증 정보는 소스 압축 파일에 넣지 않습니다. Kernel 제목은 요청한 slug에서 생성하므로 작업 생성 과정에서 실행 ID가 달라지지 않도록 맞춥니다.

실행 가능성 검사 패키지는 `--profile-only`, 처리량 검사 패키지는 `--throughput-profile`을 사용합니다. 두 모드는 함께 지정할 수 없으며 둘 다 `--stylometric-model`을 거부합니다. 어느 검사 worker도 `--complete-release`를 요청하지 않습니다. 다음 예시는 시간 측정용 입력만 준비하며 업로드나 작업 시작은 수행하지 않습니다.

```powershell
.\.venv\Scripts\python.exe -I -m scripts.package_kaggle --output .local/kaggle-throughput-package --dataset-id YOURNAME/nlu-throughput-source --kernel-id YOURNAME/nlu-large-throughput --throughput-profile
```

`nlu-source.zip`에는 정확한 실행 소스 스냅샷, 계획, 분리 명세와 라벨 CSV가 들어갑니다. 최초 Transformer 가중치, 실행 환경, Git 정보, 최초 노트북, 개인 로그와 라벨 없는 추론 입력은 제외합니다. 완료한 실행 가능성 검사 패키지는 23,780,198바이트였으며 이후 패키지는 각자의 실제 크기와 해시를 기록합니다. 별도의 작은 `stylometric-model.zip`은 `NLU-Stylometric-Model` 아래에 검증한 CPU 모델 파일 7개를 포함합니다.

Kaggle은 업로드한 압축 파일을 자동으로 풀어 제공할 수 있습니다. 외부 스크립트는 정확한 ZIP 또는 유일하게 일치하는 폴더를 받아 모든 파일의 크기와 SHA256을 확인합니다. 심볼릭 링크와 위험한 경로를 거부하고 선언한 파일만 복사합니다. 이어서 [`scripts/setup_kaggle_runtime.py`](../scripts/setup_kaggle_runtime.py)로 격리된 Python 3.12.15 CUDA 환경을 준비합니다. Kaggle 전역 환경이나 다른 프로젝트 환경을 재사용하거나 변경하지 않습니다.

## 학습과 릴리스 단계

생성된 작업은 다음 worker를 실행합니다.

```bash
python -m scripts.run_kaggle_experiment \
  --repo-root /kaggle/working/nlu-source/NLU-Authorship-Verification \
  --manifest /kaggle/working/nlu-source/NLU-Authorship-Verification/reports/split-manifest.json \
  --output /kaggle/working/nlu-results \
  --plan /kaggle/working/nlu-source/NLU-Authorship-Verification/configs/experiment-plan.json \
  --stylometric-model-dir /kaggle/working/nlu-stylometric-model/NLU-Stylometric-Model \
  --complete-release
```

여기서 `python`은 bootstrap이 선택한 격리된 CUDA 실행기입니다. Worker는 CPU 앙상블의 파일 해시, 데이터 분리, 실험 규칙과 실행 소스 연결을 확인한 뒤 대응 Transformer를 학습합니다. `--complete-release`는 임계값 0.5의 원시 손실 비교를 보존하고, 모델 선택 집합에서 각 입력 순서 정책을 정한 뒤 세 모델을 독립적으로 보정합니다. 이어서 저장한 모델 선택 logits와 고정 보정값으로 배포할 Large 파이프라인을 고릅니다. Custom ASL은 보정된 CE보다 macro F1이 0.002 이상 증가해야 채택합니다. 원시 비교와 보정된 파이프라인의 선택 모델을 구분해 보고하며, 보정 후 증가량에는 손실, 입력 순서와 보정이 함께 반영됩니다.

최종 선택에는 추가 GPU 추론이 필요하지 않으며 모델 선택 집합에서 보정값을 다시 맞추지 않습니다. 모든 최종 평가 추론 전에 세 모델과 최종 결정을 함께 고정합니다. 최종 결과로 손실, 입력 순서, temperature나 임계값을 다시 선택하지 않습니다. 이 릴리스 규칙은 실측 전체 학습의 Large 검증 결과를 관찰하기 전에 선언합니다.

`experiment-completion.json`, `training-report.json`, worker 완료 기록과 로그, 공통 종료 결정, 각 모델의 체크포인트 보고서, 폐기한 검사 기록, 실행 환경, 두 입력 검증 기록, CPU 실험 보고서와 전체 릴리스 증거를 보존합니다. 완료 조건은 `release_completed=true`, 일치하는 근거와 `experiment-failure.json`이 없는 상태입니다. Kernel의 정상 종료만으로는 충분하지 않습니다. CPU 재검증과 공개 모델의 비로그인 다운로드 확인은 학습 이후 별도로 검사합니다.

소스 패키지에는 사전에 선언한 [`bootstrap 도우미`](../scripts/bootstrap_results.py)도 포함합니다. 고정한 최종 예측이 생긴 뒤 seed 378의 대응 문서 그룹 표집 1,000회로 정확도, macro F1과 보정된 ASL에서 CE를 뺀 macro F1 차이의 조건부 95% percentile 구간을 보고합니다. 모델을 불러오거나 릴리스를 다시 학습하고 선택할 수 없습니다. 학습 seed 한 번에서 데이터 수준의 불확실성을 확인하며 반복 학습이나 새 저자 및 도메인의 일반화 불확실성을 추정하지 않습니다.
