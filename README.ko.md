# Authorship Verification

**Dongmyung Park과 Juho Kim이 2026년 2월부터 5월까지 공동 개발한 NLP 프로젝트입니다.**

[English](./README.md) | [재현 가이드](./docs/reproduction.ko.md) | [구조](./docs/architecture.ko.md) | [평가 규칙](./docs/evaluation.ko.md) | [학습](./docs/training.ko.md)

영어 글 쌍의 저자 동일성을 확인하기 위해 최초의 두 모델 계열인 문체 기반 XGBoost-LightGBM 앙상블과 RoBERTa-Large를 비교합니다. 새로운 실행 체계는 문서가 분리된 데이터, 조건을 맞춘 CE와 직접 구현한 이진 비대칭 focal 학습, 독립 보정, 변경하지 않는 평가와 해시를 확인한 CPU 재검증을 제공합니다. 서로 다른 저자(`0`) 또는 동일 저자(`1`) 라벨을 예측하며 임의의 새로운 글에서 사람의 신원을 입증하는 도구는 아닙니다.

보고서의 **custom focal은 최초 프로젝트에서 ASL이라고 부른 구현**이며, 실행기와 모델 파일의 식별자는 `custom_asl`입니다. 이미 잘 맞히는 쉬운 사례의 학습 비중을 줄이는 클래스별 비대칭 이진 focal 손실입니다. 프로젝트 고유 구현과 논문의 ASL 구현을 구분하기 위해 이 기술 명칭을 함께 사용합니다.

## 현재 릴리스 상태

새 RoBERTa-Large CE와 custom focal 학습, 학습 데이터만 사용한 제한된 문체 실험과 Kaggle 고정 평가를 완료했습니다. Large 두 개 모두 같은 6 epoch를 완료하고 5 epoch 체크포인트를 저장했습니다. 영어와 한국어 [실측 결과](./docs/results.ko.md)에 분류, 확률 보정, 입력 순서, 길이와 판정 보류 결과를 정리했습니다.

| 최종 4,708쌍 전체의 고정 파이프라인 | Macro F1 | 정확도 | 정답 |
|---|---:|---:|---:|
| 문체 XGBoost-LightGBM 앙상블 | 0.741498 | 74.1504% | 3,491 |
| RoBERTa-Large CE, 배포용 | 0.808383 | 80.8624% | 3,807 |
| RoBERTa-Large ASL(프로젝트 구현), 보존한 비교 모델 | 0.815170 | 81.5208% | 3,838 |

CE는 최종 평가 전에 보정값을 고정한 파이프라인을 선택 집합에서 비교해 결정했습니다. 이후 관찰한 focal의 최종 우위로 선택을 바꾸지 않았습니다. 최초 개발 F1 0.8348과는 평가 조건이 달라 통제된 전후 비교가 아닙니다. 보정으로 세 모델의 ECE와 NLL은 줄었지만 모든 F1이 오른 것은 아니며, 판정한 사례 정확도 90% 목표에는 미달했습니다.

[전체 고정 기록](./reports/evaluation-freeze.json)과 원본 순서의 예측을 소스, 데이터와 모델 파일 명세에 대조해 확인했습니다. 같은 실행 및 설정 파일 35개로 Windows 11에서 구현 검사 109/109, Ubuntu 24.04 WSL2에서 109/109를 통과했습니다. 모델 세 개 모두 두 환경에서 4,708쌍 전체 추론을 완료했으며, 여섯 번의 이진 분류, 판정 보류, F1과 정확도가 모두 일치했습니다. 문체 모델은 두 환경에서 정해 둔 비교 허용 오차도 충족했습니다. Large 두 개는 GPU 참조 확률값의 사전 허용 오차 `1e-5`를 넘었습니다. Windows와 Linux의 보정 확률 최대 차이는 CE가 각각 `1.79e-5`, `8.81e-6`, custom focal이 `1.80e-4`, `1.10e-4`였습니다. Linux CE의 보정 확률은 기준 안에 있었지만 원래 입력 순서의 확률 차이는 기준을 넘었습니다. 허용 오차, 참조나 보정값을 바꾸지 않고 엄격한 검증 실패로 보존했습니다. 같은 조건의 CPU 실행 비용 측정도 완료해 [CPU 요약](./reports/cpu-reproduction.json)과 [실측 결과](./docs/results.ko.md)에 연결했으며, 모델 공개와 전체 비로그인 다운로드 검증은 별도로 완료했습니다. 두 로컬 환경은 같은 물리적 컴퓨터를 사용하므로 독립된 외부 검증으로 표현하지 않습니다.

대응 학습은 실제 경과 6.3959시간, 배정 worker 합산 12.7758시간이 걸렸고, 고정 평가까지의 전체 wrapper는 실제 경과 7.5475시간입니다. 대응 학습 9시간과 전체 커널 10시간 상한을 지켰습니다. 앞선 시간 측정 가중치는 모두 폐기했으며 제한된 비교 후 GPU 품질 실험을 더 진행하지 않았습니다. 자체 배정 시간 범위와 계정의 할당량 조회는 구분하고, 공식 T4x2 사용량 배수를 검증했다고 주장하지 않습니다.

## CPU 실행

새 고정 모델 파일은 [공개 Kaggle 데이터셋 버전 1](https://www.kaggle.com/datasets/dongmyungpark/nlu-authorship-frozen-large-20261012)로 제공합니다. 기존 공개 CLI로 로그인 없이 파일 21개, 총 2,856,189,574바이트를 내려받아 고정한 크기와 SHA256이 모두 일치하는지 확인했습니다. 별도 [다운로드 검증](./reports/model-download-verification.json)은 파일 전달을 확인한 기록이며 저장 모델 추론을 다시 수행했다는 의미는 아닙니다. 저장소 최상위 폴더에서 Windows PowerShell로 실행합니다.

```powershell
.\launch-nlu.bat doctor
.\launch-nlu.bat verify-data
.\launch-nlu.bat download
.\launch-nlu.bat download --verify-only
.\launch-nlu.bat predict
```

Linux에서는 같은 명령 앞에 `sh ./launch-nlu.sh`를 사용합니다. 실행기는 프로젝트 내부에 격리된 Python 3.12 환경을 만들고 고정 CPU 패키지를 설치하며 로컬 GPU를 숨깁니다. Docker나 API key는 필요하지 않습니다. 최초 구성과 모델 다운로드에는 인터넷이 필요하고, 이후 추론은 로컬 가중치와 토크나이저를 사용합니다.

기본 모델은 CPU 문체 앙상블입니다. 선택적으로 Large를 사용하면 독립적으로 맞춘 보정값을 적용해 모델 선택 집합에서 결정한 파이프라인을 실행합니다.

```powershell
.\launch-nlu.bat download --model transformer
.\launch-nlu.bat predict --model transformer --text-file-1 passage-a.txt --text-file-2 passage-b.txt
```

검증한 릴리스 명세 [reports/artifact-download.json](./reports/artifact-download.json)은 모델 파일만 포함한 공개 Kaggle 데이터셋 버전 1의 비로그인 raw 주소와 모든 체크포인트, 토크나이저, 변환기 및 설정의 바이트 수와 SHA256을 고정합니다. 원격 파일은 `kind=file`로 원래 `models/ce/`, `models/custom_asl/`, `models/stylometric/` 실행 경로에 직접 연결합니다. 전체 비로그인 HTTP 200/해시 확인과 CLI verify-only 검사를 완료했습니다. 아래의 최초 Drive 파일은 새 릴리스의 파일 목록이나 평가 고정 기록을 대신할 수 없습니다.

모델 ZIP을 별도로 전달받은 경우에는 `models/모델명/` 구조를 같은 강화 버전 소스의 `artifacts/runtime`에 해제합니다. 실행기가 예측이나 평가 전에 설치한 파일을 전체 고정 기록에 대조하며 직접 설치에는 자동 다운로드 명세가 필요하지 않습니다. 검증을 완료한 공개 전달 경로는 위 자동 다운로드입니다.

`doctor`는 패키지, 환경 격리와 작은 CPU tensor 연산을 확인하고, `verify-data`는 CSV 해시와 문서 분리를 확인합니다. 어느 명령도 실제 모델 정확도를 측정하지 않습니다. 예측은 클래스, 보정된 동일 저자 점수와 판정 보류 시 `uncertain`을 구분합니다. 입력 제한과 전체 옵션은 [CPU 실행 가이드](./docs/runtime.ko.md)에 있습니다.

## 고정한 최종 평가 재현

릴리스 파일이 준비된 뒤 아직 없는 출력 폴더를 지정합니다.

```powershell
.\launch-nlu.bat evaluate --model stylometric --model-name stylometric --model-dir artifacts/runtime/models/stylometric --output .local/replay-stylometric --batch-size 4 --threads 4
```

두 Large 체크포인트는 다음과 같이 검사합니다.

```powershell
.\launch-nlu.bat download --model ce
.\launch-nlu.bat download --model custom_asl
.\launch-nlu.bat evaluate --model transformer --model-name ce --model-dir artifacts/runtime/models/ce --output .local/replay-ce --batch-size 4 --threads 4
.\launch-nlu.bat evaluate --model transformer --model-name custom_asl --model-dir artifacts/runtime/models/custom_asl --output .local/replay-custom-asl --batch-size 4 --threads 4
```

위 전체 재검증 예시는 배치 4와 CPU 스레드 4를 명시합니다. 새 모델의 성능 결과를 확인하기 전에 기존 모델의 작은 시간 측정으로 정한 실행 설정이며, 모델 선택이나 보정값을 바꾸지 않습니다. CLI 기본값은 배치 8과 스레드 1을 유지합니다. 최종 모델의 실제 CPU 실행 비용은 [실측 결과](./docs/results.ko.md)에 제시하며, 별도 속도 비교는 배치 1과 스레드 1을 사용했습니다.

기본 평가는 최종 글 쌍 4,708개 전체를 실행합니다. 소스, 데이터, 모델과 보정의 연결을 확인하고 다시 맞추지 않은 상태로 지표, 혼동행렬과 원본 순서의 예측을 저장합니다. 배포용으로 하나를 고르더라도 Large FP32 체크포인트 두 개를 유지합니다. 작은 예측이나 테스트 통과는 전체 저장 모델 재검증을 대신하지 않습니다. 같은 컴퓨터의 운영체제 두 개는 같은 호스트의 별도 환경으로 보고하며 독립된 외부 컴퓨터라고 표현하지 않습니다.

## 조건을 통제한 실험

제공된 학습 글 쌍 27,643개와 개발 글 쌍 5,993개를 함께 검사합니다. 전처리 후 빈 문서를 포함한 86행과 라벨이 일치하는 중복 글 쌍 2,162행을 제외해 31,388쌍을 유지합니다. 문서 동일성으로 연결한 그룹 전체를 한 집합에 배정합니다.

| 역할 | 글 쌍 수 | 용도 |
|---|---:|---|
| 학습 | 21,972 | 과제 학습과 학습 집합만의 TF-IDF |
| 모델 선택 | 3,139 | 체크포인트, 특징 및 가중치, 배포 정책 |
| 보정 | 1,569 | Temperature, 분류 및 판정 보류 임계값 |
| 최종 평가 | 4,708 | 함께 고정한 뒤 최종 평가 |

배정된 T4를 하나씩 사용하는 독립 worker 두 개가 같은 초기화, seed, 기록과 epoch 순서로 고정한 새 RoBERTa-Large를 학습합니다. CE와 최초 직접 구현한 두 클래스 softmax focal 목적함수 `-(1-p_y)^gamma_y * log(p_y)`를 비교하며 `gamma_0=2`, `gamma_1=1`을 FP32로 계산합니다. 확률 clipping을 적용한 발표된 다중 라벨 sigmoid ASL과는 다른 구현입니다.

별도 CPU 실험은 모든 TF-IDF를 학습 집합에서만 맞추고 120분 안에 제한한 XGBoost-LightGBM 설정 4개를 비교합니다. 최종 앙상블은 두 구성 모델을 모두 사용하며, 새 설정이나 가중치를 채택하려면 기준보다 모델 선택 macro F1이 0.002 이상 증가해야 합니다.

원시 AB 손실 비교는 임계값 0.5를 유지합니다. 모델 선택 집합에서 각 입력 순서를 고르고 세 모델을 독립적으로 보정합니다. 배포 결정은 저장한 모델 선택 logits에 고정 보정값을 적용한 Large 파이프라인끼리 비교하며, custom ASL이 보정된 CE보다 macro F1을 0.002 이상 높일 때만 채택합니다. 원시 비교와 배포용 선택 모델을 구분해 보고합니다. 모든 최종 추론 전에 함께 고정하고 최종 결과로 판정 기준을 조정하지 않습니다.

계획에는 seed 378의 대응 문서 그룹 bootstrap 1,000회도 있습니다. 조건부 95% percentile 구간은 고정 데이터와 학습 seed 한 번에 관한 것이며 반복 학습의 변동이나 새 저자 및 도메인의 일반화를 뜻하지 않습니다. 상세 규칙은 [평가 문서](./docs/evaluation.ko.md)에 있습니다.

## 최초 개발 기록

2026년 2월부터 5월까지의 노트북, 포스터, 예측과 모델 카드는 공동 개발자 정보를 유지합니다. 아래 수치는 최초 개발 집합의 결과이며 새 고정 평가와 구분합니다.

| 최초 모델 | 당시 보고한 개발 macro F1 | 최초 파일 다운로드 |
|---|---:|---|
| [문체 앙상블](./model-cards/stylometric-ensemble.ko.md) | 0.7702 | [최초 앙상블과 변환기 Drive](https://drive.google.com/drive/folders/1mF0bMpzfs3XODpQA2Hd7H_JvXLk5AGgp?usp=sharing) |
| [RoBERTa-Large 직접 구현 비대칭 focal](./model-cards/roberta-asymmetric-loss.ko.md) | 0.8348 | [최초 체크포인트와 토크나이저 Drive](https://drive.google.com/drive/folders/1Ty9-EpOLAgzvTWPuqrHBTbehmZZuDs3V?usp=sharing) |

개발 글 쌍 5,993개 중 630개는 학습 글 쌍과 완전히 같습니다. 이 완전 중복을 포함한 총 2,738개의 개발 글 쌍에는 학습에 이미 있는 문서가 하나 이상 포함됩니다. 그중 1,658쌍은 한 문서, 1,080쌍은 두 문서를 재사용합니다. 최초 문체 노트북은 학습과 개발 글을 합쳐 TF-IDF를 맞추므로 개발 글이 어휘와 IDF에 영향을 줍니다. 체크포인트, 앙상블 가중치와 임계값도 개발 집합에서 골랐습니다. 따라서 최초 점수와 새 최종 평가 점수의 차이는 조건을 통제한 전후 개선량이 아닙니다. 새로운 규칙은 문서 그룹을 분리하고 TF-IDF를 학습 집합에서만 맞춥니다.

[최초 노트북](./notebooks/)과 [포스터 미리보기](./docs/authorship-verification-poster-preview.png)([PDF](./docs/authorship-verification-poster.pdf?raw=1))는 공동 개발 기록입니다. 도메인 변화에 따른 과적합 감소는 독립된 도메인 분리 평가로 검증하지 않았습니다. `test.csv` 5,985쌍과 `live.csv` 20쌍에는 라벨이 없어 측정한 정확도를 제공하지 않습니다.

## 저장소 구성

```text
nlu_authorship/             데이터, 모델, 학습, 보정과 고정 평가
configs/                   사전에 선언한 실험 계획
reports/                   분리 명세, 실제 검사와 릴리스 증거
scripts/                   격리 구성, 다운로드, Kaggle 패키징과 보고
launch-nlu.bat / .sh        공개 CPU 실행 명령
docs/                      영어 기본 및 한국어 가이드와 최초 포스터
model-cards/               최초 개발 모델로 명시한 문서
notebooks/                 공동 개발한 최초 학습과 데모 노트북
results/                   최초의 라벨 없는 예측
artifacts/                 Git에서 제외한 다운로드 모델
```

저자 ID와 도메인 라벨이 없습니다. 확인한 범위는 제공된 데이터의 문서 분리이며, 과거 데이터 사용과 측정하지 않은 의미 유사 문서는 한계로 남습니다. Transformer는 두 글 합계 512토큰 이후를 자르며 짧거나 익숙하지 않은 글은 불안정할 수 있습니다.
