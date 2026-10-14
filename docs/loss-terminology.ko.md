# 손실 함수의 명칭과 구현

[English](./loss-terminology.md) | [프로젝트 개요](../README.ko.md) | [실측 결과](./results.ko.md) | [구조](./architecture.ko.md)

교차 엔트로피와 비교한 모델은 **직접 구현한 클래스별 비대칭 이진 focal 손실**로 학습했습니다. 최초 프로젝트에서는 이를 `ASL` 또는 `ASLoss`라고 불렀고, 공개 실행기의 호환 식별자는 `custom_asl`로 유지합니다. 이 이름은 보존한 자료를 연결하기 위한 것으로, 공식 ASL 구현을 학습하거나 재현했다는 의미는 아닙니다.

## 실제로 구현한 방식

서로 다른 저자는 클래스 `0`, 동일 저자는 클래스 `1`입니다. 두 클래스 softmax에서 정답 클래스의 확률을 `p_y`라고 하면, `gamma_0=2`, `gamma_1=1`을 적용합니다.

`L = mean(-(1 - p_y)^gamma_y * log(p_y))`

Focal 가중치는 이미 잘 맞히는 쉬운 사례의 학습 기여를 줄이고, 클래스 `0`에서 더 강하게 줄입니다. 비대칭 focusing 아이디어를 참고해 클래스별 가중치를 정한 구현이며, 새로운 손실 함수의 제안이나 ASL 논문의 재현으로 설명하지 않습니다.

[최초 노트북](../notebooks/roberta-asymmetric-loss/training.ipynb)은 클래스를 `ASLoss`라고 명명하고 softmax 확률에 `log(max(p_y, 1e-8))`를 적용합니다. [현재 구현](../nlu_authorship/losses.py)은 FP32 log-softmax로 정답 클래스의 로그 확률을 직접 계산합니다. Focal 원리와 클래스별 지수는 유지하면서 기존 로그 계산의 확률 하한을 제거했습니다. 따라서 하한보다 작은 확률에서는 값이 달라질 수 있으며, 극단적인 확률이나 혼합 정밀도에서 수치적으로 완전히 같다는 뜻은 아닙니다.

## 공식 ASL과의 차이

논문 저자의 [공식 구현](https://github.com/Alibaba-MIIL/ASL/blob/main/src/loss_functions/losses.py)에는 다중 라벨 버전과 단일 라벨 버전이 모두 있습니다. ASL을 다중 라벨 문제에만 사용할 수 있다고 설명하는 것은 정확하지 않습니다.

| 구현 | 확률과 가중치의 구성 |
|---|---|
| 공식 다중 라벨 `AsymmetricLoss` | 독립 sigmoid 출력, 양성과 음성의 비대칭 focusing, 설정 가능한 음성 확률 이동 |
| 공식 `ASLSingleLabel` | Softmax, 정답과 비정답 클래스 마스크에 따른 가중치, 설정 가능한 label smoothing |
| 이 프로젝트의 직접 구현 이진 focal | 두 클래스 softmax, 실제 정답 클래스에 따른 지수(`gamma_0=2`, `gamma_1=1`), 음성 확률 이동과 label smoothing 없음 |

이 프로젝트의 코드는 두 공식 구현 중 어느 것과도 같지 않습니다. 두 클래스 softmax를 사용한다는 사실만으로 `ASLSingleLabel`과 같아지는 것은 아닙니다. [ASL 논문](https://openaccess.thecvf.com/content/ICCV2021/html/Ridnik_Asymmetric_Loss_for_Multi-Label_Classification_ICCV_2021_paper.html)은 비대칭 focusing 아이디어의 배경 자료입니다.

## 학습과 배포 선택

조건을 맞춘 RoBERTa-Large 실험에서는 CE와 이 직접 구현 이진 focal 목적함수를 비교했습니다. 두 모델 모두 6 epoch를 완료하고 5 epoch 체크포인트를 저장했습니다. 고정한 선택 파이프라인의 macro F1은 CE `0.801199`, 직접 구현 focal `0.799145`였습니다. Focal은 사전에 정한 `+0.002` 교체 기준에 도달하지 않아 CE를 배포 모델로 유지했습니다.

최종 4,708쌍 전체에서 보존한 focal 파이프라인의 macro F1은 `0.815170`, 정확도는 `81.5208%`(`3,838 / 4,708`)였습니다. 고정한 CE 파이프라인은 `0.808383`, `80.8624%`(`3,807 / 4,708`)였습니다. 이후의 최종 평가 결과로 배포 모델을 다시 선택하지 않았습니다. 이 차이는 입력 순서와 보정 선택까지 포함한 전체 파이프라인의 결과이며, 손실 함수만의 인과 효과나 보편적인 우월성을 입증하지 않습니다.

## 보존한 식별자와 기록

최초 노트북 이름, `ASLoss`, `Models_ASL` 경로와 실행기 및 모델의 `custom_asl` 식별자는 호환성과 개발 이력 확인을 위해 유지합니다. 고정한 설정, 가중치, 예측과 성능 보고서도 변경하지 않습니다. 독자가 보는 설명에는 **직접 구현 비대칭 이진 focal(기존 명칭: ASL)**을 사용해 공식 ASL과 구분합니다. 명칭을 바로잡는 작업으로 재학습하거나 결과를 다시 계산하지 않습니다.
