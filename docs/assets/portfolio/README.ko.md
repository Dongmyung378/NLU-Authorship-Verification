# 포트폴리오 시각 자료

[English](README.md) | [한국어](README.ko.md)

보존한 한국어 SVG를 바탕으로 한 표시용 파생본입니다. 수치는 `visual-data.json`에 기록한 기존 실험 자료와 체크섬을 근거로 합니다. 영문 문서는 같은 데이터 도형과 팔레트 및 수치를 유지한 영어 파생본을 연결합니다. 개념 썸네일의 두 언어 사본에서는 글 A와 B를 함께 모델 계열에 연결하도록 설명 화살표만 명시적 예외로 보완했습니다. 원본 SVG와 데이터 그래프는 그대로입니다. 공동 개발의 영문 원본 포스터와 노트북, 학습 모델과 데이터 분할, 선택 판단 및 원래 실험 보고서는 변경하지 않습니다.

생성기는 Python 기본 라이브러리와 포함된 표시용 데이터만 사용합니다. 그래프를 다시 생성하는 것은 모델 실행이나 과학적 평가 검증이 아닙니다.

그래프의 **Large 커스텀 ASL**은 최초 프로젝트에서 `ASLoss`로 이름 붙인 직접 구현 손실을 사용합니다. 정답 클래스의 softmax 확률에 `-(1-p_y)^gamma_y * log(p_y)`를 적용하며, 다른 저자는 `gamma_0=2`, 동일 저자는 `gamma_1=1`입니다. **ASL의 비대칭 focusing을 적용한 커스텀 이진 focal 변형이며, 공식 ASL 또는 `ASLSingleLabel` 코드의 기본 설정을 그대로 재현한 실험과 구분합니다.** 두 로짓 차이에 단일 sigmoid를 적용해 같은 이진 확률을 표현할 수 있으므로, softmax 사용만으로 ASL 계열이 아니라고 구분하지 않습니다. 호환성을 위해 실행 식별자 `custom_asl`은 보존합니다. [공식 손실 함수 구현](https://github.com/Alibaba-MIIL/ASL/blob/main/src/loss_functions/losses.py)과 [손실 함수 명칭 안내](../../loss-terminology.ko.md)에서 설정과 수치 처리의 차이를 확인할 수 있습니다. 이번 수정은 명칭과 설명에 한정하며, 표시한 수치와 그래프의 도형 및 색상은 그대로 유지합니다.

```bash
python docs/assets/portfolio/render_visuals.py --input docs/assets/portfolio/visual-data.json --output docs/assets/portfolio
```

`visual-receipt.json`은 보존한 한국어 SVG를 표시용 데이터에 연결합니다. 영어 파생본은 아래 명령으로 생성하고 대조합니다. `localization.json`에 문구를, `localization-receipt.json`에 원본 및 파생본의 SHA-256을 기록합니다. 이 작업은 기존 사이트와 ZIP을 변경하거나 성능 측정을 다시 수행하지 않습니다.

```bash
python docs/assets/portfolio/render_localized_visuals.py
python docs/assets/portfolio/render_localized_visuals.py --check
```

## 시각 자료

- [글 쌍과 모델 계열](nlu-thumbnail.ko.svg)
- [nlu-results.svg](nlu-results.svg)
- [nlu-calibration.svg](nlu-calibration.svg)
