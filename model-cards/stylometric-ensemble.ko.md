# 문체 기반 XGBoost-LightGBM 앙상블

[English](./stylometric-ensemble.md) · [프로젝트 개요](../README.ko.md)

## 모델 요약

두 영어 글이 같은 저자에 의해 작성되었는지 예측하는 지도학습 이진 분류 모델입니다. 단어·문자·기능어 TF-IDF와 직접 설계한 문체 특징을 결합하고, XGBoost와 LightGBM의 확률을 최적화된 가중치와 임계값으로 앙상블합니다.

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

여러 도메인에서 수집된 영어 텍스트 쌍 27,643개를 사용했습니다. 외부 학습 데이터는 추가하지 않았습니다. 학습·개발·라벨 없는 테스트 데이터는 저장소의 [data](../data/)에 있습니다.

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

### LightGBM

| 파라미터 | 값 |
|---|---:|
| Estimators | 2,000; 1,700번째 반복에서 조기 종료 |
| Learning rate | 0.01 |
| Leaves | 127 |
| Subsample | 0.8 |
| Column sample by tree | 0.8 |
| Max bins | 127 |

### 앙상블

| 설정 | 값 |
|---|---:|
| XGBoost 가중치 | 0.26 |
| LightGBM 가중치 | 0.74 |
| 분류 임계값 | 0.43 |

## 평가

다른 저자 쌍 2,937개와 동일 저자 쌍 3,056개로 구성된 개발 데이터 5,993개를 평가했습니다.

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
- [학습된 모델 파일 다운로드](https://drive.google.com/drive/folders/1mF0bMpzfs3XODpQA2Hd7H_JvXLk5AGgp?usp=sharing)

## 한계와 위험

- 영어에 맞춰 설계되어 다국어 또는 코드 스위칭 입력을 안정적으로 처리하지 못합니다.
- 다섯 단어보다 짧은 글에서는 문체 특징의 정보량이 크게 줄어듭니다.
- TF-IDF 학습 어휘에 없는 단어는 반영되지 않습니다.
- 장르가 달라지면 문체 패턴도 바뀌어 정확도가 낮아질 수 있습니다.
- 저자 판별은 원치 않는 익명 해제에 사용될 수 있으므로 신원을 확정하는 단독 근거로 사용하면 안 됩니다.

## 재현

[학습 노트북](../notebooks/stylometric-ensemble/training.ipynb), [데모 노트북](../notebooks/stylometric-ensemble/demo.ipynb), [한국어 재현 가이드](../docs/reproduction.ko.md)를 참고하세요.
