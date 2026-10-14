# 직접 구현 이진 ASL(클래스별 비대칭 focal)을 적용한 RoBERTa-Large

[English](./roberta-asymmetric-loss.md) | [프로젝트 개요](../README.ko.md)

이 카드는 최초 RoBERTa-Large 개발 모델을 설명합니다. 성능 수치는 당시 개발 데이터의 결과이며, 문서를 분리하고 고정한 사전학습 가중치에서 새로 시작하는 RoBERTa-Large 실험과 조건을 통제한 비교 결과가 아닙니다.

## 모델 요약

제공된 영어 텍스트 쌍 데이터에서 동일 저자와 다른 저자 라벨을 분류하도록 학습한 지도학습 이진 분류 모델입니다. <code>roberta-large</code>에 시퀀스 분류 헤드를 추가하고 직접 구현한 이진 비대칭 focal loss로 파인튜닝했습니다. 보고된 성능은 제공된 개발 데이터에 한정됩니다.

- **개발자:** Dongmyung Park, Juho Kim
- **언어:** 영어
- **과업:** 텍스트 쌍 기반 저자 동일성 판별
- **구조:** 이진 분류 헤드를 적용한 RoBERTa-Large
- **기반 모델:** [FacebookAI/roberta-large](https://huggingface.co/FacebookAI/roberta-large)

## 입력 표현

토큰화 전에 이메일 헤더, 이메일 주소, URL, 불필요한 공백을 제거합니다. 두 글을 RoBERTa의 표준 텍스트 쌍 형식으로 인코딩하고 <code>longest_first</code> 방식으로 전체 길이를 최대 512토큰까지 제한합니다.

기본 교차 엔트로피 대신 두 클래스의 softmax 확률에 `-(1-p_y)^gamma_y * log(max(p_y, 1e-8))`를 적용하고 예시별 손실을 평균합니다. <code>gamma_neg=2</code>, <code>gamma_pos=1</code>을 사용해 쉬운 예시의 기여를 낮추며, 클래스 0에 더 큰 지수를 적용합니다. 최초 노트북은 이 클래스를 `ASLoss`라고 불렀지만, 실제로는 직접 구현한 클래스별 비대칭 이진 focal 손실입니다. 공식 다중 라벨 ASL과 단일 라벨 `ASLSingleLabel`은 모두 존재합니다. 이 코드는 ASL의 이진 변형을 직접 구현했으며, 공식 기본 설정이나 논문의 실험을 그대로 재현한 것으로 설명하지 않습니다. `1e-8` 하한은 로그 계산의 수치 안정성 처리이며 ASL의 음성 확률 이동이 아닙니다. [손실 함수의 명칭과 비교](../docs/loss-terminology.ko.md)에서 공식 방식과 현재 FP32 log-softmax의 차이를 설명합니다. 최초 노트북 이름과 경로는 개발 이력으로 보존합니다. 개발 데이터의 임계값 탐색으로 추론 임계값 <code>0.51</code>을 선택했습니다. 도메인이 바뀌었을 때 일반화 성능을 높인다는 인과 관계는 검증되지 않았습니다.

## 학습 데이터

제공된 [`train.csv`](../data/train.csv)의 영어 텍스트 쌍 27,643개로 학습했으며 이 과제의 학습을 위한 라벨 있는 쌍은 추가하지 않았습니다. 학습과 개발 CSV에는 글과 쌍의 라벨만 있고 저자 ID나 도메인 라벨은 없으므로, 이 파일만으로 저자나 도메인을 구분한 분리 평가 여부를 확인할 수 없습니다.

## 하이퍼파라미터

| 파라미터 | 값 |
|---|---:|
| Learning rate | 5e-6 |
| 학습 배치 크기 | 32 |
| 평가 배치 크기 | 32 |
| 최대 시퀀스 길이 | 512 |
| 최대 epoch | 14 |
| 저장된 최적 체크포인트 | Epoch 10, step 8,640 |
| 조기 종료 대기 | 평가 epoch 3회 |
| Warmup ratio | 0.1 |
| Weight decay | 0.01 |
| Loss | 직접 구현한 이진 비대칭 focal loss (<code>gamma_neg=2</code>, <code>gamma_pos=1</code>) |
| 분류 임계값 | 0.51 |

내려받은 체크포인트의 `trainer_state.json`에는 epoch 10.0, global step 8,640, best metric 0.8346138164975799가 기록돼 있습니다. 이전 카드의 epoch 13 종료 주장은 이 체크포인트의 상태 파일로 확인되지 않습니다. 선택된 최적 체크포인트와 학습이 최종 중단된 시점은 서로 다른 정보입니다.

## 평가

다른 저자 쌍 2,937개와 동일 저자 쌍 3,056개로 구성된 개발 데이터 5,993개를 평가했습니다. 제공된 CSV를 대조한 결과, 개발 데이터 630쌍(10.5%)이 학습 데이터의 텍스트 쌍과 순서까지 정확히 같고 라벨도 일치했습니다. 모델 체크포인트와 분류 임계값도 이 개발 데이터에서 선택했습니다. 따라서 아래 수치는 개발 데이터 결과이며, 독립적인 새 저자나 도메인에서의 성능을 입증하지 않습니다.

| 지표 | 점수 |
|---|---:|
| Macro F1 | **0.8348** |
| 동일 저자 F1 | 0.8365 |
| 정확도 | 0.83 |
| Macro precision | 0.83 |
| Macro recall | 0.83 |

추가 분석에서는 저장된 체크포인트와 노트북의 전처리, 저장된 임계값 0.51로 같은 `data/dev.csv`를 다시 평가했습니다.

| 개발 데이터 하위 집합 | 쌍 수 | Macro F1 |
|---|---:|---:|
| 두 글 합계 50단어 이하 | 554 | 0.6680 |
| 두 글 합계 100단어 초과 | 4,775 | 0.8599 |
| 학습 데이터와 텍스트 쌍이 정확히 같은 경우 | 630 | 0.9857 |
| 정확히 같은 텍스트 쌍을 제외한 경우 | 5,363 | 0.8170 |

단어 수는 공백 기준으로 계산했습니다. 이 값들은 동일한 개발 데이터를 사후에 나누어 본 결과이며, 구간이 서로 겹칠 수 있습니다. 독립적인 테스트 결과나 새 도메인 성능 추정치는 아닙니다.

클래스별 결과:

| 클래스 | 정밀도 | 재현율 | F1 |
|---|---:|---:|---:|
| 다른 저자 | 0.83 | 0.84 | 0.83 |
| 동일 저자 | 0.84 | 0.83 | 0.84 |

## 연산 환경과 모델 파일

- 학습 시간: 약 6시간 11분
- 학습 하드웨어: NVIDIA A100 40GB GPU 1개
- 모델 파라미터: 약 3억 5,500만 개
- 모델 가중치: 1,421,495,360바이트, 약 1.42 GB 또는 1.32 GiB
- 추론: 시스템 메모리 16GB와 CUDA 지원 GPU 및 CUDA 지원 PyTorch 빌드 권장, CPU에서도 가능하지만 속도가 느림
- [학습된 모델 파일 다운로드](https://drive.google.com/drive/folders/1Ty9-EpOLAgzvTWPuqrHBTbehmZZuDs3V?usp=sharing). 내려받은 모델 파일은 GitHub에 올리지 않으며 저장소에서 `models/`를 무시합니다.

## 다운로드한 모델 실행

저장소 최상위 폴더에서 위 링크의 파일을 모두 내려받아 `models/Models_ASL/`(원래 다운로드 폴더 이름) 또는 `models/roberta-asymmetric-loss/` 바로 아래에 놓습니다. 해당 폴더에는 `best_threshold.json`, `config.json`, `model.safetensors`, `tokenizer.json`, `tokenizer_config.json`이 있어야 합니다. 함께 내려받은 추가 토크나이저 파일도 같은 폴더에 보관하고, 선택한 폴더와 이 파일들 사이에 폴더가 한 단계 더 들어가지 않도록 합니다.

[최초 터미널 시연용 의존성](../requirements-inference.txt)을 설치합니다. 아래 예시는 CPU 사용을 명시하며 영어 글 두 개를 입력받습니다. 최초 GPU 학습과 시연 절차는 아래에 연결한 노트북에 보존합니다. 현재 [재현 가이드](../docs/reproduction.ko.md)는 새로운 고정 CPU 릴리스를 설명하며 이 과거 모델의 CUDA 구성 안내가 아닙니다.

```bash
python -m pip install -r requirements-inference.txt
python verify_author.py --model roberta --device cpu
```

기본값인 `--device auto`는 CUDA를 사용할 수 있으면 GPU를 선택합니다. NVIDIA GPU가 감지되었지만 PyTorch에서 CUDA를 사용할 수 없다면 설정 방법을 안내하며 중단하고, NVIDIA GPU가 감지되지 않으면 CPU를 사용합니다. GPU 사용을 필수로 하려면 `--device cuda`, CPU로 강제하려면 `--device cpu`를 지정하세요. 명령줄에서 바로 입력하려면 `python verify_author.py --model roberta --text-1 "First English passage" --text-2 "Second English passage"`를 실행합니다. 출력에는 `Same-author probability`, 모델 파일의 분류 임계값, 판별 결과가 표시됩니다. 표시 명칭과 달리 이 값은 클래스 1의 원시 softmax 점수이며, 두 글의 저자가 같을 확률로 보정된 값은 아닙니다. 임의의 새 저자나 도메인에서의 성능은 검증되지 않았습니다. 다른 위치에 모델을 두었다면 파일이 직접 들어 있는 폴더를 `--model-dir PATH`로 지정합니다. [현재 한국어 재현 가이드](../docs/reproduction.ko.md)는 새 고정 CPU 릴리스를 설명합니다. 해당 릴리스의 모델 파일과 평가 연결은 이 과거 시연과 구분합니다.

## 한계와 위험

- 두 입력의 합이 512토큰을 넘으면 잘리므로 유용한 정보가 손실될 수 있습니다.
- 매우 짧은 글의 결과는 불안정할 수 있습니다. 학습 데이터의 글 길이 중앙값은 공백 기준 약 123단어로, 짧은 터미널 예시보다 훨씬 깁니다.
- 기반 모델과 파인튜닝 데이터가 영어 중심이어서 다국어, 코드 스위칭, 방언이 많은 입력은 신뢰하기 어렵습니다.
- 학습 비용이 높고 배포 모델의 크기도 큽니다.
- 장르나 도메인이 바뀌면 정확도가 낮아질 수 있습니다.
- 저자 판별은 원치 않는 익명 해제에 사용될 수 있으므로 신원을 확정하는 단독 근거로 사용하면 안 됩니다.

## 재현

[학습 노트북](../notebooks/roberta-asymmetric-loss/training.ipynb), [데모 노트북](../notebooks/roberta-asymmetric-loss/demo.ipynb), [한국어 재현 가이드](../docs/reproduction.ko.md)를 참고하세요.
