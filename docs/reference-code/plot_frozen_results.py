"""Render the unchanged frozen plot with a project palette and readable labels.

Usage: python docs/reference-code/plot_frozen_results.py --summary ... --output ...
The frozen scripts/plot_results.py remains byte-identical. This display-only
wrapper keeps values, text, support counts, ticks and the existing n=5 padding.
No model package, checkpoint, image inputs or inference code is imported.
"""
import hashlib
import json
from pathlib import Path
import runpy

from matplotlib.figure import Figure

MODEL_COLORS = ["#998B78", "#484552", "#963F55"]


def plotted_information(figure):
    """Capture content, coordinates and axes independently of color/type style."""
    return [{"title": axis.get_title(), "xlabel": axis.get_xlabel(), "ylabel": axis.get_ylabel(),
             "xlim": list(axis.get_xlim()), "ylim": list(axis.get_ylim()),
             "xticks": list(axis.get_xticks()), "yticks": list(axis.get_yticks()),
             "xticklabels": [text.get_text() for text in axis.get_xticklabels()],
             "bars": [[p.get_x(), p.get_y(), p.get_width(), p.get_height()] for p in axis.patches],
             "annotations": [[text.get_text(), list(text.get_position())] for text in axis.texts],
             "legend": [text.get_text() for text in axis.get_legend().get_texts()] if axis.get_legend() else []}
            for axis in figure.axes]


def style_paired_bars(axis):
    if len(axis.patches) != 6:
        raise ValueError("Expected the original three paired model comparisons")
    for index, patch in enumerate(axis.patches):
        patch.set_facecolor(MODEL_COLORS[index % 3])
        patch.set_alpha(.48 if index < 3 else .98)
        patch.set_edgecolor("white")
        patch.set_linewidth(.6)
    legend = axis.get_legend()
    if legend:
        for index, handle in enumerate(legend.legend_handles):
            handle.set_facecolor("#998B78")
            handle.set_alpha(.48 if index == 0 else .98)
            handle.set_edgecolor("white")


def main():
    frozen_plot = Path(__file__).resolve().parents[2] / "scripts" / "plot_results.py"
    save_original = Figure.savefig

    def save_with_label_room(figure, *args, **kwargs):
        if len(figure.axes) != 3:
            raise ValueError("The frozen evaluation figure must have exactly three panels")
        middle = figure.axes[1]
        middle.set_ylim(0, 1.12)
        middle.set_yticks([0, .2, .4, .6, .8, 1.0])
        before = plotted_information(figure)
        style_paired_bars(figure.axes[0])
        style_paired_bars(figure.axes[2])
        for patch in middle.patches:
            patch.set_facecolor("#484552")
            patch.set_edgecolor("white")
            patch.set_linewidth(.6)
        figure.set_facecolor("#FDFBFC")
        for axis in figure.axes:
            axis.set_facecolor("white")
            axis.spines[["left", "bottom"]].set_color("#B6ADB2")
            axis.tick_params(colors="#484552", labelsize=10)
            axis.grid(axis="y", color="#EDE6E9", alpha=1, linewidth=.75)
            axis.title.set_color("#963F55")
            axis.title.set_fontsize(13)
            axis.title.set_fontweight("semibold")
            axis.xaxis.label.set_color("#484552")
            axis.yaxis.label.set_color("#484552")
            axis.xaxis.label.set_fontsize(10.5)
            axis.yaxis.label.set_fontsize(10.5)
            for text in axis.texts:
                text.set_color("#3D3440")
            legend = axis.get_legend()
            if legend:
                legend.get_frame().set_edgecolor("#DED2D7")
                legend.get_frame().set_facecolor("white")
                for text in legend.get_texts():
                    text.set_color("#484552")
        for text in figure.texts:
            text.set_color("#963F55")
        if before != plotted_information(figure):
            raise ValueError("Styling changed graph information or numerical coordinates")
        print(json.dumps({"display_only": True, "plotted_information_unchanged": True,
                          "plotted_information_sha256": hashlib.sha256(json.dumps(before, sort_keys=True, default=lambda value: value.item()).encode()).hexdigest(),
                          "model_colors": dict(zip(("stylometric", "ce", "custom_asl"), MODEL_COLORS))}))
        return save_original(figure, *args, **kwargs)

    Figure.savefig = save_with_label_room
    try:
        runpy.run_path(str(frozen_plot), run_name="__main__")
    finally:
        Figure.savefig = save_original


if __name__ == "__main__":
    main()
