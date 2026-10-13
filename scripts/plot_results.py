"""Plot measured fixed-holdout outcomes from the machine-readable release."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--summary', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    report = json.loads(args.summary.read_text(encoding='utf-8'))
    if report.get('status') != 'completed' or report.get('final_outcomes_used_for_selection') is not False:
        raise ValueError('A completed frozen release report is required')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    names = ['stylometric', 'ce', 'custom_asl']
    labels = ['Stylometric', 'RoBERTa CE', 'RoBERTa focal']
    models = [report['models'][name] for name in names]
    counts = {item['count'] for item in models}
    if len(counts) != 1 or any(item['freeze_sha256'] != report['freeze_sha256'] for item in models):
        raise ValueError('Plot comparisons must share denominator and freeze')
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    colors = ['#65758d', '#277f99', '#219f91']
    x = np.arange(3)
    axes[0].bar(x - .18, [item['raw_ab_metrics']['macro_f1'] for item in models], .36, color=colors, alpha=.55, label='Raw AB at0.5')
    axes[0].bar(x + .18, [item['macro_f1'] for item in models], .36, color=colors, label='Frozen calibrated policy')
    axes[0].set_xticks(x, labels, rotation=15)
    axes[0].set_ylabel('Macro F1')
    axes[0].set_ylim(0, 1)
    axes[0].legend(fontsize=9)
    for index, model in enumerate(models):
        axes[0].text(index + .18, model['macro_f1'] + .025, f"{model['macro_f1']:.4f}", ha='center')
    axes[0].set_title('Matched final evaluation')
    selected = report['models'][report['selected_transformer']]
    bins = ['under_50', '50_to_199', '200_plus']
    quality = [selected['length_bins'][name] for name in bins]
    values = [item['macro_f1'] if item['count'] else np.nan for item in quality]
    axes[1].bar(range(3), values, color='#277f99')
    axes[1].set_xticks(range(3), ['<50', '50-199', '>=200'])
    axes[1].set_xlabel('Minimum cleaned words in either passage')
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel('Macro F1')
    for index, item in enumerate(quality):
        axes[1].text(index, values[index] + .025 if item['count'] else .025,
                     f"n={item['count']:,}" if item['count'] else 'No samples', ha='center', fontsize=10)
    axes[1].set_title('Selected transformer by input length')
    raw = [item['raw_metrics']['ece10'] for item in models]
    calibrated = [item['ece10'] for item in models]
    axes[2].bar(x - .18, raw, .36, color='#e8a556', label='Before calibration')
    axes[2].bar(x + .18, calibrated, .36, color='#219f91', label='Frozen temperature')
    axes[2].set_xticks(x, labels, rotation=15)
    axes[2].set_ylabel('ECE (fixed10bins; lower is better)')
    axes[2].legend(fontsize=9)
    axes[2].set_title('Calibration evaluated on holdout')
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.15)
        ax.set_axisbelow(True)
    fig.suptitle(f"NLU document-disjoint release | {counts.pop():,} final pairs | seed378", fontsize=18, weight='bold')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError('Preserve previous figure')
    fig.savefig(args.output, dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
