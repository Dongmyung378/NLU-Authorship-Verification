# 포트폴리오 시각 자료

[English](README.md) | [한국어](README.ko.md)

사이트에서 사용하는 한국어 SVG 그래프와 썸네일입니다. 수치는 `visual-data.json`에 기록한 기존 실험 자료와 체크섬을 근거로 합니다. 학습 모델과 데이터 분할, 선택 판단 및 원래 실험 보고서는 변경하지 않습니다. 평가 범위와 한계는 프로젝트의 본문 문서를 확인하십시오.

생성기는 Python 기본 라이브러리와 포함된 표시용 데이터만 사용합니다. 그래프를 다시 생성하는 것은 모델 실행이나 과학적 평가 검증이 아닙니다.

그래프의 **Large 커스텀 focal**은 최초 프로젝트에서 `ASLoss`로 이름 붙인 직접 구현 손실을 사용합니다. 정답 클래스의 softmax 확률에 `-(1-p_y)^gamma_y * log(p_y)`를 적용하며, 다른 저자는 `gamma_0=2`, 동일 저자는 `gamma_1=1`입니다. **ASL은 당시 프로젝트에서 사용한 명칭이며, 공식 다중 라벨 ASL이나 단일 레이블 `ASLSingleLabel`을 그대로 구현한 것은 아닙니다.** 호환성을 위해 실행 식별자 `custom_asl`은 보존합니다. 구분의 근거는 [공식 손실 함수 구현](https://github.com/Alibaba-MIIL/ASL/blob/main/src/loss_functions/losses.py)에서 확인할 수 있습니다. 이번 수정은 명칭과 설명에 한정하며, 표시한 수치와 그래프의 도형 및 색상은 그대로 유지합니다.

```bash
python docs/assets/portfolio/render_visuals.py --input docs/assets/portfolio/visual-data.json --output docs/assets/portfolio
```

`visual-receipt.json`은 SVG 파일의 바이트 수와 체크섬을 표시용 데이터에 연결합니다. 소스 다운로드 ZIP에도 사이트와 같은 시각 자료를 포함합니다. 처리 흐름도는 구현한 단계를 설명하며 새로운 성능 측정값이 아닙니다.

## 시각 자료

- [nlu-thumbnail.svg](nlu-thumbnail.svg)
- [nlu-results.svg](nlu-results.svg)
- [nlu-calibration.svg](nlu-calibration.svg)
