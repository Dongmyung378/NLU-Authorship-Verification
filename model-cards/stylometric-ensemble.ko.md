# 문체 기반 XGBoost-LightGBM 앙상블

[English](./stylometric-ensemble.md) · [프로젝트 개요](../README.ko.md)

## 모델 요약

제공된 영어 텍스트 쌍 데이터에서 동일 저자와 다른 저자 라벨을 분류하도록 학습한 지도학습 이진 분류 모델입니다. 단어·문자·기능어 TF-IDF와 직접 설계한 문체 특징을 결합하고, XGBoost와 LightGBM의 출력을 최적화된 가중치와 임계값으로 앙상블합니다. 보고된 성능은 제공된 개발 데이터에 한정됩니다.

- **개발자:** Dongmyung Park, Juho Kim
- **언어:** 영어
- **과업:** 텍스트 쌍 기반 저자 동일성 판별
- **구조:** XGBoost + LightGBM 가중 앙상블
- **학습 방식:** 라벨이 있는 텍스트 쌍으로 처음부터 학습

## 입력 표현

파이프라인은 다음 특징을 사용합니다.

- 최대 7,000개 특징의 단어 1–2 gram TF-IDF
- 최대 7,000개 특징의 문자 2–4 gram TF-IDF
- 고정된 영어 어휘를 사용하는 기능어 TF-IDF
- 길이 통계, 구두점, 대문자 사용, 어휘 다양성, 접미사 패턴을 포함한 문체 특징 24개
- 두 글 사이의 절댓값 차이, 원소별 곱, 비율, 평균, 코사인 거리

전처리 과정에서는 이메일, URL, 날짜, 전화번호, 길게 반복된 문자, 반복 구두점, 불필요한 공백을 제거합니다.

## 학습 데이터

제공된 [`train.csv`](../data/train.csv)의 영어 텍스트 쌍 27,643개로 학습했으며 이 과제의 학습을 위한 라벨 있는 쌍은 추가하지 않았습니다. 학습·개발 CSV에는 글과 쌍의 라벨만 있고 저자 ID나 도메인 라벨은 없으므로, 이 파일만으로 저자별·도메인별 분리 평가 여부를 확인할 수 없습니다.

## 하이퍼파라미터

### XGBoost

| 파라미터 | 값 |
|---|---:|
| Estimators | 7,000; 6,213번째 반복에서 조기 종료 |
| Learning rate | 0.01 |
| Max depth | 7 |
| Subsample | 0.8 |
| Column sample by tree | 0.8 |
| Minimum child weight | 2 |
| Max bins | 128 |
| 조기 종료 대기 | 200회 반복 |

### LightGBM

| 파라미터 | 값 |
|---|---:|
| Estimators | 2,000; 1,700번째 반복에서 조기 종료 |
| Learning rate | 0.01 |
| Leaves | 127 |
| Subsample | 0.8 |
| Subsample frequency | 1 |
| Column sample by tree | 0.8 |
| Max bins | 127 |
| 조기 종료 대기 | 100회 반복 |

### 앙상블

| 설정 | 값 |
|---|---:|
| XGBoost 가중치 | 0.26 |
| LightGBM 가중치 | 0.74 |
| 분류 임계값 | 0.43 |

## 평가

다른 저자 쌍 2,937개와 동일 저자 쌍 3,056개로 구성된 개발 데이터 5,993개를 평가했습니다. 제공된 CSV를 대조한 결과, 개발 데이터 630쌍(10.5%)이 학습 데이터의 텍스트 쌍과 순서까지 정확히 같고 라벨도 일치했습니다. 조기 종료, 앙상블 가중치, 분류 임계값도 이 개발 데이터를 사용했습니다. 따라서 아래 수치는 개발 데이터 결과이며, 독립적인 새 저자나 도메인에서의 성능을 입증하지 않습니다.

| 지표 | 점수 |
|---|---:|
| Macro F1 | **0.7702** |
| 동일 저자 F1 | 0.7813 |
| 정확도 | 0.77 |
| Macro precision | 0.77 |
| Macro recall | 0.77 |

클래스별 결과:

| 클래스 | 정밀도 | 재현율 | F1 |
|---|---:|---:|---:|
| 다른 저자 | 0.78 | 0.74 | 0.76 |
| 동일 저자 | 0.76 | 0.80 | 0.78 |

## 연산 환경과 모델 파일

- 학습 시간: 약 1시간 10분
- 학습 하드웨어: NVIDIA T4 16GB GPU 2개와 멀티코어 CPU
- 추론: 일반 CPU 지원, 시스템 메모리 약 2GB 권장
- 직렬화된 앙상블·벡터라이저·설정 파일: 총 60MB 미만
- [학습된 모델 파일 다운로드](https://drive.google.com/drive/folders/1mF0bMpzfs3XODpQA2Hd7H_JvXLk5AGgp?usp=sharing). 내려받은 모델 파일은 GitHub에 올리지 않으며 저장소에서 `models/`를 무시합니다.

## 다운로드한 모델 실행

저장소 최상위 폴더에서 위 링크의 파일을 모두 내려받아 `models/Models_ML/`(원래 다운로드 폴더 이름) 또는 `models/stylometric-ensemble/` 바로 아래에 놓습니다. 해당 폴더에는 `ensemble_config.json`, `xgb_model.json`, `lgb_model.pkl`, `tfidf_word.pkl`, `tfidf_char.pkl`, `tfidf_fw.pkl`이 있어야 합니다. 선택한 폴더와 이 파일들 사이에 폴더가 한 단계 더 들어가지 않도록 합니다.

[터미널 시연용 의존성](../requirements-inference.txt)을 설치하고 영어 글 두 개를 입력합니다.

```bash
python -m pip install -r requirements-inference.txt
python verify_author.py --model stylometric
```

명령줄에서 바로 입력하려면 `python verify_author.py --model stylometric --text-1 "First English passage" --text-2 "Second English passage"`를 실행합니다. 이 파이프라인은 CPU에서 실행됩니다. 출력에는 `Same-author probability`, 모델 파일의 분류 임계값, 판별 결과가 표시됩니다. 표시 명칭과 달리 이 값은 두 분류기의 출력을 결합한 점수이며, 실제 동일 저자 확률로 보정되지 않았습니다. 임의의 새 저자나 도메인에서의 성능은 검증되지 않았습니다. 다른 위치에 모델을 두었다면 파일이 직접 들어 있는 폴더를 `--model-dir PATH`로 지정합니다. 환경 설정과 노트북 사용법은 [한국어 재현 가이드](../docs/reproduction.ko.md)를 참고하세요.

## 한계와 위험

- 영어에 맞춰 설계되어 다국어 또는 코드 스위칭 입력을 안정적으로 처리하지 못합니다.
- 다섯 단어보다 짧은 글에서는 문체 특징의 정보량이 크게 줄어듭니다.
- 짧은 글의 결과는 불안정할 수 있습니다. 학습 데이터의 글 길이 중앙값은 공백 기준 약 123단어로, 짧은 터미널 예시보다 훨씬 깁니다.
- TF-IDF 학습 어휘에 없는 단어는 반영되지 않습니다.
- 장르가 달라지면 문체 패턴도 바뀌어 정확도가 낮아질 수 있습니다.
- 저자 판별은 원치 않는 익명 해제에 사용될 수 있으므로 신원을 확정하는 단독 근거로 사용하면 안 됩니다.

## 재현

[학습 노트북](../notebooks/stylometric-ensemble/training.ipynb), [데모 노트북](../notebooks/stylometric-ensemble/demo.ipynb), [한국어 재현 가이드](../docs/reproduction.ko.md)를 참고하세요.
