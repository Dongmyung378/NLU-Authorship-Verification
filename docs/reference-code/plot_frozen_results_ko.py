"""Render a Korean-label derivative of the retained English frozen figure.

Use the same --summary and --output arguments as plot_frozen_results.py. The
original plotting code, report, English PNG, bar coordinates, limits, counts
and palette remain unchanged. Only the new output's labels and font change.
Requires Matplotlib and an installed supported Korean font. Malgun Gothic is
preferred when available; missing fonts stop rendering rather than falling
back to a font that cannot display Korean.
"""
from pathlib import Path
import re
import runpy
from matplotlib.figure import Figure
from matplotlib.text import Text
from matplotlib import font_manager

KOREAN_FONTS = (
    'Malgun Gothic', 'Noto Sans CJK KR', 'Noto Sans KR',
    'Apple SD Gothic Neo', 'NanumGothic',
)

LABELS = {
    'Stylometric': '문체 앙상블', 'RoBERTa CE': 'RoBERTa CE',
    'RoBERTa focal': 'RoBERTa focal', 'Raw AB at0.5': '원래 AB, 임계값 0.5',
    'Frozen calibrated policy': '고정한 보정 파이프라인',
    'Matched final evaluation': '동일한 최종 평가',
    'Minimum cleaned words in either passage': '짧은 쪽 글의 전처리 후 단어 수',
    'Selected transformer by input length': '선택한 Large CE의 입력 길이별 성능',
    'Before calibration': '보정 전', 'Frozen temperature': '고정한 온도 보정',
    'ECE (fixed10bins; lower is better)': 'ECE (고정 10구간, 작을수록 좋음)',
    'Calibration evaluated on holdout': '최종 평가의 확률 보정 비교',
    'No samples': '표본 없음',
}


def select_korean_font(manager=None):
    """Use installed Matplotlib font metadata and require a real font file."""
    if manager is None:
        manager = font_manager.fontManager
    installed = {entry.name for entry in manager.ttflist if Path(entry.fname).is_file()}
    for family in KOREAN_FONTS:
        if family in installed:
            return family
    raise RuntimeError(
        'No supported Korean font is installed. Install one of: '
        + ', '.join(KOREAN_FONTS)
        + '. Rendering stopped before writing a figure.'
    )


def main():
    korean_font = select_korean_font()
    original_save = Figure.savefig

    def save_korean(figure, *args, **kwargs):
        geometry = [[(p.get_x(), p.get_y(), p.get_width(), p.get_height())
                     for p in axis.patches] for axis in figure.axes]
        for text in figure.findobj(Text):
            original = text.get_text()
            translated = LABELS.get(original, original)
            title = re.fullmatch(r'NLU document-disjoint release \| ([\d,]+) final pairs \| seed378', original)
            if title:
                translated = f'문서 분리 NLU 평가 | 최종 {title[1]}쌍 | 시드378'
            text.set_text(translated)
            text.set_fontfamily(korean_font)
        for axis in (figure.axes[0], figure.axes[2]):
            axis.set_xticks(axis.get_xticks(), ['문체 앙상블', 'RoBERTa CE', 'RoBERTa focal'], rotation=15)
            for text in axis.get_xticklabels():
                text.set_fontfamily(korean_font)
        if geometry != [[(p.get_x(), p.get_y(), p.get_width(), p.get_height())
                         for p in axis.patches] for axis in figure.axes]:
            raise ValueError('Localization changed plotted measurements')
        return original_save(figure, *args, **kwargs)

    Figure.savefig = save_korean
    try:
        runpy.run_path(str(Path(__file__).with_name('plot_frozen_results.py')), run_name='__main__')
    finally:
        Figure.savefig = original_save


if __name__ == '__main__':
    main()
