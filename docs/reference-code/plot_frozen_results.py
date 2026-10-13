"""Render the frozen scientific plot with room for its measured n=5 label.

Usage: python docs/reference-code/plot_frozen_results.py --summary ... --output ...
This calls the unchanged frozen plot helper. It adjusts only middle-panel
display padding before save; values, support counts and F1 ticks stay fixed.
"""
from pathlib import Path
import runpy

from matplotlib.figure import Figure


def main():
    frozen_plot = Path(__file__).resolve().parents[2] / "scripts" / "plot_results.py"
    save_original = Figure.savefig

    def save_with_label_room(figure, *args, **kwargs):
        if len(figure.axes) != 3:
            raise ValueError("The frozen evaluation figure must have exactly three panels")
        middle = figure.axes[1]
        middle.set_ylim(0, 1.12)
        middle.set_yticks([0, .2, .4, .6, .8, 1.0])
        return save_original(figure, *args, **kwargs)

    Figure.savefig = save_with_label_room
    try:
        runpy.run_path(str(frozen_plot), run_name="__main__")
    finally:
        Figure.savefig = save_original


if __name__ == "__main__":
    main()
