# Asymmetric Loss를 적용한 RoBERTa-Large

[English](./roberta-asymmetric-loss.md) · [프로젝트 개요](../README.ko.md)

## 모델 요약

두 영어 글이 같은 저자에 의해 작성되었는지 예측하는 지도학습 이진 분류 모델입니다. <code>roberta-large</code>에 시퀀스 분류 헤드를 추가하고 Asymmetric Loss로 파인튜닝해, 문체 차이가 미묘한 다른 저자 쌍을 더 집중적으로 학습합니다.

- **개발자:** Dongmyung Park, Juho Kim
- **언어:** 영어
- **과업:** 텍스트 쌍 기반 저자 동일성 판별
- **구조:** 이진 분류 헤드를 적용한 RoBERTa-Large
- **기반 모델:** [FacebookAI/roberta-large](https://huggingface.co/FacebookAI/roberta-large)

## 입력 표현

토큰화 전에 이메일 헤더, 이메일 주소, URL, 불필요한 공백을 제거합니다. 두 글을 RoBERTa의 표준 텍스트 쌍 형식으로 인코딩하고 <code>longest_first</code> 방식으로 전체 길이를 최대 512토큰까지 제한합니다.

기본 교차 엔트로피 대신 <code>gamma_neg=2</code>, <code>gamma_pos=1</code>인 Asymmetric Loss를 사용합니다. 쉬운 예시의 영향을 낮추면서 다른 저자 클래스에 더 강한 초점 패널티를 적용합니다. 개발 데이터의 임계값 탐색을 통해 추론 임계값 <code>0.51</code>을 선택했습니다.

## 학습 데이터

여러 도메인에서 수집된 영어 텍스트 쌍 27,643개를 사용했습니다. 외부 학습 데이터는 추가하지 않았습니다. 학습·개발·라벨 없는 테스트 데이터는 저장소의 [data](../data/)에 있습니다.

## 하이퍼파라미터

| 파라미터 | 값 |
|---|---:|
| Learning rate | 5e-6 |
| 학습 배치 크기 | 32 |
| 평가 배치 크기 | 32 |
| 최대 시퀀스 길이 | 512 |
| Epochs | 14; 13번째 epoch에서 조기 종료 |
| 조기 종료 대기 | 평가 epoch 3회 |
| Warmup ratio | 0.1 |
| Weight decay | 0.01 |
| Loss | Asymmetric Loss (<code>gamma_neg=2</code>, <code>gamma_pos=1</code>) |
| 분류 임계값 | 0.51 |

## 평가

다른 저자 쌍 2,937개와 동일 저자 쌍 3,056개로 구성된 개발 데이터 5,993개를 평가했습니다. 분류 임계값도 같은 개발 데이터에서 선택했으므로 아래 수치는 독립적인 테스트 세트 성능이 아닌 개발 데이터 성능입니다.

| 지표 | 점수 |
|---|---:|
| Macro F1 | **0.8348** |
| 동일 저자 F1 | 0.8365 |
| 정확도 | 0.83 |
| Macro precision | 0.83 |
| Macro recall | 0.83 |

클래스별 결과:

| 클래스 | 정밀도 | 재현율 | F1 |
|---|---:|---:|---:|
| 다른 저자 | 0.83 | 0.84 | 0.83 |
| 동일 저자 | 0.84 | 0.83 | 0.84 |

## 연산 환경과 모델 파일

- 학습 시간: 약 6시간 11분
- 학습 하드웨어: NVIDIA A100 40GB GPU 1개
- 모델 파라미터: 약 3억 5,500만 개
- 모델 가중치: 약 1.32GB
- 추론: 시스템 메모리 16GB와 CUDA 지원 GPU 권장, CPU에서도 가능하지만 속도가 느림
- [학습된 모델 파일 다운로드](https://drive.google.com/drive/folders/1Ty9-EpOLAgzvTWPuqrHBTbehmZZuDs3V?usp=sharing). 내려받은 모델 파일은 GitHub에 올리지 않으며 저장소에서 `models/`를 무시합니다.

## 다운로드한 모델 실행

저장소 최상위 폴더에서 위 링크의 파일을 모두 내려받아 `models/Models_ASL/`(원래 다운로드 폴더 이름) 또는 `models/roberta-asymmetric-loss/` 바로 아래에 놓습니다. 해당 폴더에는 `best_threshold.json`, `config.json`, `model.safetensors`, `tokenizer.json`, `tokenizer_config.json`이 있어야 합니다. 함께 내려받은 추가 토크나이저 파일도 같은 폴더에 보관하고, 선택한 폴더와 이 파일들 사이에 폴더가 한 단계 더 들어가지 않도록 합니다.

[터미널 시연용 의존성](../requirements-inference.txt)을 설치하고 영어 글 두 개를 입력합니다.

```bash
python -m pip install -r requirements-inference.txt
python verify_author.py --model roberta
```

명령줄에서 바로 입력하려면 `python verify_author.py --model roberta --text-1 "First English passage" --text-2 "Second English passage"`를 실행합니다. 출력에는 동일 저자 확률, 모델 파일의 분류 임계값, 판별 결과가 표시됩니다. 다른 위치에 모델을 두었다면 파일이 직접 들어 있는 폴더를 `--model-dir PATH`로 지정합니다. CPU에서도 추론할 수 있지만 시간이 더 걸릴 수 있어 CUDA 지원 GPU를 권장합니다. 환경 설정과 노트북 사용법은 [한국어 재현 가이드](../docs/reproduction.ko.md)를 참고하세요.

## 한계와 위험

- 두 입력의 합이 512토큰을 넘으면 잘리므로 유용한 정보가 손실될 수 있습니다.
- 기반 모델과 파인튜닝 데이터가 영어 중심이어서 다국어, 코드 스위칭, 방언이 많은 입력은 신뢰하기 어렵습니다.
- 학습 비용이 높고 배포 모델의 크기도 큽니다.
- 장르나 도메인이 바뀌면 정확도가 낮아질 수 있습니다.
- 저자 판별은 원치 않는 익명 해제에 사용될 수 있으므로 신원을 확정하는 단독 근거로 사용하면 안 됩니다.

## 재현

[학습 노트북](../notebooks/roberta-asymmetric-loss/training.ipynb), [데모 노트북](../notebooks/roberta-asymmetric-loss/demo.ipynb), [한국어 재현 가이드](../docs/reproduction.ko.md)를 참고하세요.
