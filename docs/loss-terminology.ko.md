# 손실 함수의 명칭과 구현

[English](./loss-terminology.md) | [프로젝트 개요](../README.ko.md) | [실측 결과](./results.ko.md) | [구조](./architecture.ko.md)

교차 엔트로피와 비교한 모델은 **직접 구현한 ASL의 이진 변형인 클래스별 비대칭 focal 손실**로 학습했습니다. 최초 프로젝트에서는 이를 `ASL` 또는 `ASLoss`라고 불렀고, 보고서에는 custom focal로 표기합니다. 공개 실행기의 호환 식별자는 `custom_asl`로 유지합니다. ASL 계열의 변형이며, 공식 라이브러리의 기본 설정이나 논문의 실험을 그대로 재현했다는 의미는 아닙니다.

## 실제로 구현한 방식

서로 다른 저자는 클래스 `0`, 동일 저자는 클래스 `1`입니다. 두 클래스 softmax에서 정답 클래스의 확률을 `p_y`라고 하면, `gamma_0=2`, `gamma_1=1`을 적용합니다.

`L = mean(-(1 - p_y)^gamma_y * log(p_y))`

Focal 가중치는 이미 잘 맞히는 쉬운 사례의 학습 기여를 줄이고, 클래스 `0`에서 더 강하게 줄입니다. [Torchvision의 sigmoid focal loss](https://docs.pytorch.org/vision/main/generated/torchvision.ops.sigmoid_focal_loss.html)에서도 gamma를 사용하므로, gamma가 있다는 사실만으로 ASL을 구분할 수는 없습니다. 이진 양성과 음성 클래스에 서로 다른 지수를 적용한 부분이 이 변형의 비대칭 focusing입니다. 새로운 손실 함수의 제안이나 논문의 실험 재현으로 설명하지 않습니다.

[최초 노트북](../notebooks/roberta-asymmetric-loss/training.ipynb)은 클래스를 `ASLoss`라고 명명하고 softmax 확률에 `log(max(p_y, 1e-8))`를 적용합니다. [현재 구현](../nlu_authorship/losses.py)은 FP32 log-softmax로 정답 클래스의 로그 확률을 직접 계산합니다. Focal 원리와 클래스별 지수는 유지하면서 기존 로그 계산의 확률 하한을 제거했습니다. 따라서 하한보다 작은 확률에서는 값이 달라질 수 있으며, 극단적인 확률이나 혼합 정밀도에서 수치적으로 완전히 같다는 뜻은 아닙니다.

## 공식 ASL 버전과의 관계

논문 저자의 [공식 구현](https://github.com/Alibaba-MIIL/ASL/blob/main/src/loss_functions/losses.py)에는 다중 라벨 버전과 단일 라벨 버전이 모두 있습니다. ASL을 다중 라벨 문제에만 사용할 수 있다고 설명하는 것은 정확하지 않습니다.

| 구현 | 확률과 가중치의 구성 |
|---|---|
| 공식 다중 라벨 `AsymmetricLoss` | 독립 sigmoid 출력, 양성과 음성의 비대칭 focusing, 설정 가능한 음성 확률 이동 |
| 공식 `ASLSingleLabel` | Softmax, 정답과 비정답 클래스 마스크에 따른 가중치, 설정 가능한 label smoothing |
| 이 프로젝트의 직접 구현 이진 focal | 두 클래스 softmax, 실제 정답 클래스에 따른 지수(`gamma_0=2`, `gamma_1=1`), 음성 확률 이동과 label smoothing 없음 |

공식 ASL의 음성 확률 이동은 설정으로 끌 수 있으므로, 이 기능이 없다는 사실만으로 ASL 변형이 아니라고 판단할 수는 없습니다. 이진 문제에서 `d = z_1 - z_0`이면 `sigmoid(d) = softmax([z_0, z_1])_1`입니다. 공식 `AsymmetricLoss`를 `clip=0`, `gamma_neg=2`, `gamma_pos=1`, `disable_torch_grad_focal_loss=False`로 설정하고 단일 이진 출력을 사용하면, 클래스별 focal 목적함수가 이 프로젝트의 수식과 일치합니다. 다만 공식 구현의 합계와 이 프로젝트의 평균 집계, 극단적인 확률에서의 로그 하한은 구분해야 합니다. 집계 방식을 맞추고 로그 하한에 걸리지 않는 범위에서는 수학적인 목적함수가 같습니다. 수식과 공식 코드를 비교해 얻은 설명이며, 새로운 학습 실험이나 실행 결과의 비트 단위 일치를 뜻하지 않습니다.

이 프로젝트는 공식 손실을 가져와 호출하는 대신 해당 이진 변형을 직접 구현했습니다. 사용한 설정과 실험은 공식 기본 설정 및 `ASLSingleLabel`의 정답 마스크와 label smoothing 구성과 구분합니다. [ASL 논문](https://openaccess.thecvf.com/content/ICCV2021/html/Ridnik_Asymmetric_Loss_for_Multi-Label_Classification_ICCV_2021_paper.html)은 방법의 배경 자료입니다. 두 클래스 softmax나 gamma의 존재만으로 발표된 설정을 정확히 재현했다고 볼 수는 없습니다.

## 학습과 배포 선택

조건을 맞춘 RoBERTa-Large 실험에서는 CE와 이 직접 구현 이진 focal 목적함수를 비교했습니다. 두 모델 모두 6 epoch를 완료하고 5 epoch 체크포인트를 저장했습니다. 고정한 선택 파이프라인의 macro F1은 CE `0.801199`, 직접 구현 focal `0.799145`였습니다. Focal은 사전에 정한 `+0.002` 교체 기준에 도달하지 않아 CE를 배포 모델로 유지했습니다.

최종 4,708쌍 전체에서 보존한 focal 파이프라인의 macro F1은 `0.815170`, 정확도는 `81.5208%`(`3,838 / 4,708`)였습니다. 고정한 CE 파이프라인은 `0.808383`, `80.8624%`(`3,807 / 4,708`)였습니다. 이후의 최종 평가 결과로 배포 모델을 다시 선택하지 않았습니다. 이 차이는 입력 순서와 보정 선택까지 포함한 전체 파이프라인의 결과이며, 손실 함수만의 인과 효과나 보편적인 우월성을 입증하지 않습니다.

## 보존한 식별자와 기록

최초 노트북 이름, `ASLoss`, `Models_ASL` 경로와 실행기 및 모델의 `custom_asl` 식별자는 호환성과 개발 이력 확인을 위해 유지합니다. 고정한 설정, 가중치, 예측과 성능 보고서도 변경하지 않습니다. 독자가 보는 설명에는 **직접 구현 비대칭 이진 focal(ASL의 이진 변형)**을 사용해 ASL이라는 이름과 실제 목적함수를 연결하고, 공식 기본 설정의 정확한 재현이라는 주장과 구분합니다. 명칭을 바로잡는 작업으로 재학습하거나 결과를 다시 계산하지 않습니다.
