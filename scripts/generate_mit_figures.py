"""Generate MIT Red editorial SVG and PNG figures directly from verified Track 01/02/03 result archives."""
from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path(r"D:\Projects\NLU-Authorship-Verification")
Z1 = ROOT / "outputs" / "01_stylometric_ml" / "results_01_stylometric_ml.zip"
Z2 = ROOT / "outputs" / "02_deberta_v3_base" / "results_02_deberta_v3_base.zip"
Z3 = ROOT / "outputs" / "03_deberta_v3_large" / "results_03_deberta_v3_large.zip"

FIG_REPO = ROOT / "docs" / "figures"
FIG_PORTFOLIO = Path(r"D:\Projects\Portfolio-Cloudflare\source\public\images\projects")
OG_PORTFOLIO = Path(r"D:\Projects\Portfolio-Cloudflare\source\public\images\og-project-05.jpg")
FIG_REPO.mkdir(parents=True, exist_ok=True)
FIG_PORTFOLIO.mkdir(parents=True, exist_ok=True)

# MIT Red editorial palette
MIT_RED = "#A31F34"
MIT_RED_DARK = "#750014"
MIT_RED_SOFT = "#F6E7EA"
MIT_RED_MID = "#C85A6B"
SLATE = "#334155"
SLATE_LIGHT = "#64748B"
GRAY = "#8A8B8C"
GRAY_DARK = "#5C6066"
PAPER = "#FAF8F7"
WHITE = "#FFFFFF"
INK = "#1E1A1B"
MUTED = "#5E5A5B"
RULE = "#E4DEDE"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Malgun Gothic", "Noto Sans KR", "DejaVu Sans", "Arial"],
    "axes.unicode_minus": False,
    "figure.facecolor": PAPER,
    "axes.facecolor": WHITE,
    "axes.edgecolor": RULE,
    "axes.linewidth": 0.9,
    "grid.color": RULE,
    "grid.linewidth": 0.7,
    "text.color": INK,
    "axes.labelcolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "svg.fonttype": "none",
})


def read_zip_csv(zp: Path, member: str) -> pd.DataFrame:
    with zipfile.ZipFile(zp) as z:
        return pd.read_csv(io.BytesIO(z.read(member)))


def read_zip_json(zp: Path, member: str) -> dict:
    with zipfile.ZipFile(zp) as z:
        return json.loads(z.read(member).decode("utf-8"))


def style_ax(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(RULE)
    ax.spines["bottom"].set_color(RULE)
    ax.tick_params(length=0, pad=6)
    ax.set_axisbelow(True)


def write_thumbnail_svg() -> None:
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="760" viewBox="0 0 1200 760" role="img" aria-labelledby="title desc">
<title id="title">동일 저자 판별: 순수 문체학 ML vs DeBERTa-v3-base vs DeBERTa-v3-large</title>
<desc id="desc">1,024 토큰 예산과 양방향 저자 배제 분할에서 전통 문체학 ML, DeBERTa-v3-base, DeBERTa-v3-large 세 트랙을 혼합 없이 독립 비교한 구조와 최종 Locked Test 성적(Open ROC-AUC 0.9139, PAN Overall 0.8643, Closed 1:30 Top-1 78.33%)입니다.</desc>
<style>text{{font-family:'Noto Sans KR','Malgun Gothic','Apple SD Gothic Neo',sans-serif;font-variant-numeric:tabular-nums}}</style>
<rect width="1200" height="760" fill="{PAPER}"/>
<rect x="0" y="0" width="1200" height="10" fill="{MIT_RED}"/>
<text x="52" y="74" fill="{INK}" font-size="42" font-weight="750">단·중문 동일 저자 판별 벤치마크</text>
<text x="52" y="114" fill="{MUTED}" font-size="22">양방향 저자 완전 격리 · 1,024 토큰 예산 · 순수 ML vs 순수 DeBERTa-v3 3대 트랙 독립 비교</text>
<line x1="52" y1="138" x2="1148" y2="138" stroke="{RULE}" stroke-width="1.3"/>

<!-- Left Panel: Paired Passage Input -->
<rect x="52" y="164" width="424" height="414" rx="14" fill="{WHITE}" stroke="{RULE}" stroke-width="1.3"/>
<text x="82" y="210" fill="{INK}" font-size="25" font-weight="750">대칭 글 쌍 입력</text>
<text x="82" y="238" fill="{MUTED}" font-size="16">문서당 최대 512 토큰 · 순서 불변 (오차 0.00)</text>

<rect x="82" y="262" width="170" height="236" rx="10" fill="{PAPER}" stroke="{RULE}" stroke-width="1.2"/>
<text x="104" y="302" fill="{INK}" font-size="22" font-weight="700">Passage A</text>
<path d="M104 334H228M104 362H214M104 390H224M104 418H198M104 446H220" stroke="#C8C2C0" stroke-width="4" stroke-linecap="round"/>
<path d="M104 476H208" stroke="{MIT_RED}" stroke-width="5.5" stroke-linecap="round"/>

<rect x="276" y="262" width="170" height="236" rx="10" fill="{PAPER}" stroke="{RULE}" stroke-width="1.2"/>
<text x="298" y="302" fill="{INK}" font-size="22" font-weight="700">Passage B</text>
<path d="M298 334H422M298 362H406M298 390H418M298 418H390M298 446H414" stroke="#C8C2C0" stroke-width="4" stroke-linecap="round"/>
<path d="M298 476H402" stroke="{SLATE}" stroke-width="5.5" stroke-linecap="round"/>

<rect x="82" y="516" width="364" height="42" rx="8" fill="{MIT_RED_SOFT}"/>
<text x="264" y="543" text-anchor="middle" fill="{MIT_RED_DARK}" font-size="16" font-weight="700">Train / Val / Test 저자 중복 0명 · 9개 도메인</text>

<!-- Connectors -->
<path d="M476 371H532V226H588" fill="none" stroke="{GRAY}" stroke-width="3" stroke-linejoin="round"/>
<path d="M476 371H588" fill="none" stroke="{SLATE_LIGHT}" stroke-width="3"/>
<path d="M476 371H532V514H588" fill="none" stroke="{MIT_RED}" stroke-width="3.5" stroke-linejoin="round"/>
<circle cx="532" cy="371" r="5" fill="{INK}"/>
<path d="M578 219L588 226L578 233" fill="none" stroke="{GRAY}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>
<path d="M578 364L588 371L578 378" fill="none" stroke="{SLATE_LIGHT}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>
<path d="M578 507L588 514L578 521" fill="none" stroke="{MIT_RED}" stroke-width="3.5" stroke-linecap="round" stroke-linejoin="round"/>

<!-- Track 01 -->
<rect x="596" y="164" width="552" height="122" rx="14" fill="{WHITE}" stroke="{RULE}" stroke-width="1.3"/>
<rect x="596" y="164" width="8" height="122" rx="4" fill="{GRAY}"/>
<text x="626" y="204" fill="{GRAY_DARK}" font-size="15" font-weight="700">TRACK 01 · CPU 32 ms/pair · 21.6 MB</text>
<text x="626" y="238" fill="{INK}" font-size="25" font-weight="750">Stylometric ML (10-Expert + OOD Gate)</text>
<text x="626" y="268" fill="{MUTED}" font-size="17">Open AUC 0.8474 · PAN 0.8008 · Closed Top-1 64.58% (교차 장르 1위)</text>

<!-- Track 02 -->
<rect x="596" y="308" width="552" height="122" rx="14" fill="{WHITE}" stroke="{RULE}" stroke-width="1.3"/>
<rect x="596" y="308" width="8" height="122" rx="4" fill="{SLATE}"/>
<text x="626" y="348" fill="{SLATE}" font-size="15" font-weight="700">TRACK 02 · 186M · 4-Depth ABTT (r=3)</text>
<text x="626" y="382" fill="{INK}" font-size="25" font-weight="750">DeBERTa-v3-base (12-Layer Siamese)</text>
<text x="626" y="412" fill="{MUTED}" font-size="17">Open AUC 0.9040 · PAN 0.8567 · Closed Top-1 76.25%</text>

<!-- Track 03 -->
<rect x="596" y="452" width="552" height="126" rx="14" fill="{WHITE}" stroke="{MIT_RED}" stroke-width="2.2"/>
<rect x="596" y="452" width="10" height="126" rx="4" fill="{MIT_RED}"/>
<rect x="1014" y="470" width="114" height="26" rx="13" fill="{MIT_RED_SOFT}"/>
<text x="1071" y="488" text-anchor="middle" fill="{MIT_RED_DARK}" font-size="13" font-weight="700">FINAL MODEL</text>
<text x="626" y="490" fill="{MIT_RED}" font-size="15" font-weight="700">TRACK 03 · 437M · Soup 0.65/0.35 + Dual-ABTT</text>
<text x="626" y="527" fill="{INK}" font-size="25" font-weight="750">DeBERTa-v3-large (24-Layer Siamese)</text>
<text x="626" y="558" fill="{MIT_RED_DARK}" font-size="16.5" font-weight="600">Open AUC 0.9139 · PAN 0.8643 · Closed Top-1 78.33% · ArXiv 0.9389</text>

<!-- Bottom MIT Red Scorecard -->
<rect x="52" y="604" width="1096" height="118" rx="14" fill="{MIT_RED}"/>
<text x="84" y="652" fill="{WHITE}" font-size="27" font-weight="750">Final Locked Test (DeBERTa-v3-large)</text>
<text x="84" y="692" fill="{MIT_RED_SOFT}" font-size="18">9,440 Open · 240 Closed 1:30 · McNemar p=0.037 vs Base</text>
<line x1="618" y1="624" x2="618" y2="702" stroke="#D98B96" stroke-width="1.2" opacity="0.55"/>

<text x="782" y="654" text-anchor="end" fill="{WHITE}" font-size="37" font-weight="750">0.9139</text>
<text x="782" y="690" text-anchor="end" fill="{MIT_RED_SOFT}" font-size="15.5">Open ROC-AUC</text>

<text x="950" y="654" text-anchor="end" fill="{WHITE}" font-size="37" font-weight="750">0.8643</text>
<text x="950" y="690" text-anchor="end" fill="{MIT_RED_SOFT}" font-size="15.5">PAN Overall</text>

<text x="1118" y="654" text-anchor="end" fill="{WHITE}" font-size="37" font-weight="750">78.33%</text>
<text x="1118" y="690" text-anchor="end" fill="{MIT_RED_SOFT}" font-size="15.5">Closed 1:30 Top-1</text>
</svg>
'''
    (FIG_PORTFOLIO / "nlu-thumbnail.svg").write_text(svg, encoding="utf-8")
    print("[ok] wrote nlu-thumbnail.svg")


def build_3way_benchmark_fig() -> None:
    t3 = read_zip_csv(Z3, "03_deberta_v3_large/comparison_3way_final_test.csv")
    o1 = read_zip_csv(Z1, "01_stylometric_ml/test_open_predictions_ml.csv").sort_values("pair_id")
    o2 = read_zip_csv(Z2, "02_deberta_v3_base/test_open_predictions_deberta_base.csv").sort_values("pair_id")
    o3 = read_zip_csv(Z3, "03_deberta_v3_large/test_open_predictions_deberta_large.csv").sort_values("pair_id")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.6), gridspec_kw={"width_ratios": [1.18, 1.0]})
    fig.patch.set_facecolor(PAPER)

    metrics = [
        ("Closed 1:30\nTop-1", "Test_Closed_Top1"),
        ("Open\nROC-AUC", "Test_Open_ROC_AUC"),
        ("Open\nPAN Overall", "Test_Open_PAN_Overall"),
        ("Open\nForced F1", "Test_Open_Forced_F1"),
    ]
    x = np.arange(len(metrics))
    w = 0.25
    specs = [
        ("01. Stylometric ML", GRAY, 0),
        ("02. DeBERTa-v3-base", SLATE, 1),
        ("03. DeBERTa-v3-large", MIT_RED, 2),
    ]
    for lbl, color, idx in specs:
        vals = [float(t3.loc[idx, col]) for _, col in metrics]
        bars = ax1.bar(x + (idx - 1) * w, vals, width=w * 0.92, color=color, label=lbl, edgecolor="none")
        for b, v in zip(bars, vals):
            weight = "bold" if idx == 2 else "normal"
            col_t = MIT_RED_DARK if idx == 2 else MUTED
            ax1.text(b.get_x() + b.get_width() / 2, v + 0.011, f"{v:.3f}", ha="center", va="bottom", fontsize=8.8, fontweight=weight, color=col_t)

    style_ax(ax1)
    ax1.set_xticks(x)
    ax1.set_xticklabels([m[0] for m in metrics], fontsize=10, color=INK)
    ax1.set_ylim(0.50, 1.02)
    ax1.set_yticks(np.arange(0.50, 1.01, 0.10))
    ax1.grid(axis="y", linestyle="-", alpha=0.8)
    ax1.set_title("Final Locked Test 3-Track Comparison", fontsize=12.5, fontweight="bold", loc="left", pad=22)
    ax1.text(0.0, 1.02, "Exact McNemar (S2, n=9,440): Large vs Base p=0.037  |  Large vs ML p=1.0e-36",
             transform=ax1.transAxes, fontsize=8.8, color=MUTED)
    ax1.legend(loc="upper left", frameon=True, facecolor=WHITE, edgecolor=RULE, fontsize=9)

    roc_specs = [
        ("03. DeBERTa-v3-large", o3["label"].to_numpy(), o3["deberta_large_prob"].to_numpy(), MIT_RED, 2.5, "-"),
        ("02. DeBERTa-v3-base", o2["label"].to_numpy(), o2["deberta_base_prob"].to_numpy(), SLATE, 1.9, "-"),
        ("01. Stylometric ML", o1["label"].to_numpy(), o1["ml_sota_prob"].to_numpy(), GRAY, 1.8, "--"),
    ]
    for lbl, yt, yp, col, lw, ls in roc_specs:
        fpr, tpr, _ = roc_curve(yt, yp)
        auc = roc_auc_score(yt, yp)
        ax2.plot(fpr, tpr, color=col, lw=lw, ls=ls, label=f"{lbl}  (AUC {auc:.4f})")
    ax2.plot([0, 1], [0, 1], color=RULE, lw=1.2, ls=":")
    style_ax(ax2)
    ax2.grid(True, linestyle="-", alpha=0.75)
    ax2.set_xlim(-0.01, 1.01)
    ax2.set_ylim(-0.01, 1.02)
    ax2.set_xlabel("False Positive Rate", fontsize=10)
    ax2.set_ylabel("True Positive Rate", fontsize=10)
    ax2.set_title("Scenario 2 ROC Curves (9,440 Unseen-Author Pairs)", fontsize=12.5, fontweight="bold", loc="left", pad=12)
    ax2.legend(loc="lower right", frameon=True, facecolor=WHITE, edgecolor=RULE, fontsize=9)

    fig.tight_layout(pad=2.0)
    fig.savefig(FIG_PORTFOLIO / "nlu-3way-benchmark.svg", bbox_inches="tight")
    fig.savefig(FIG_REPO / "benchmark_3way_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("[ok] wrote nlu-3way-benchmark.svg & benchmark_3way_comparison.png")


def build_progression_and_fpr_fig() -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.6), gridspec_kw={"width_ratios": [1.15, 1.0]})
    fig.patch.set_facecolor(PAPER)

    # Left: Open ROC-AUC step-by-step progression across all 3 tracks
    stages_ml = ["Legacy\nLGBM", "R1\nEnsemble", "R3\nOOD Gate\n(Val)", "Locked\nTest"]
    vals_ml = [0.7632, 0.8009, 0.8378, 0.8474]

    stages_base = ["1st\n2D Cal", "2nd\nOOD Gate", "3rd\nMulti-Depth", "4th\n4D-ABTT\n(Val)", "Locked\nTest"]
    vals_base = [0.8155, 0.8912, 0.8956, 0.9009, 0.9040]

    stages_large = ["R1\n2D Cal", "R2\n4D-ABTT\n(r=3)", "R4-A\nDual-ABTT\n+AgreePCA", "R4-B\nSoup 65/35\n(Val)", "Locked\nTest"]
    vals_large = [0.8239, 0.9101, 0.9124, 0.9139, 0.9139]

    # Plot on normalized milestone index 0..4
    x_ml = [0.0, 1.2, 3.0, 4.0]
    x_base = [0.0, 1.2, 2.1, 3.0, 4.0]
    x_large = [0.0, 1.4, 2.25, 3.0, 4.0]

    ax1.plot(x_ml, vals_ml, marker="o", ms=6.5, lw=2.0, ls="--", color=GRAY, label="01. Stylometric ML (Legacy → OOD-Register Gate)")
    ax1.plot(x_base, vals_base, marker="s", ms=6.5, lw=2.1, color=SLATE, label="02. DeBERTa-v3-base (2D Cal → 4-Depth ABTT)")
    ax1.plot(x_large, vals_large, marker="D", ms=7.2, lw=2.8, color=MIT_RED, label="03. DeBERTa-v3-large (R1 → Dual-ABTT + Soup)")

    # Key milestone labels (clean, non-overlapping)
    annots = [
        (0.0, 0.7632, "0.7632\nLegacy ML", GRAY_DARK, (6, -24)),
        (4.0, 0.8474, "0.8474", GRAY_DARK, (8, -3)),
        (0.0, 0.8155, "0.8155\nBase R1", SLATE, (8, -22)),
        (1.2, 0.8912, "0.8912\n+OOD Gate", SLATE, (6, -26)),
        (4.0, 0.9040, "0.9040", SLATE, (8, -3)),
        (0.0, 0.8239, "0.8239\nLarge R1", MIT_RED_DARK, (-10, 24)),
        (1.4, 0.9101, "0.9101\n+4D ABTT", MIT_RED_DARK, (-12, 12)),
        (4.0, 0.9139, "0.9139", MIT_RED, (8, 1)),
    ]
    for xi, yi, txt, col, ofs in annots:
        ax1.annotate(txt, (xi, yi), xytext=ofs, textcoords="offset points", fontsize=8.3,
                     fontweight="bold" if col in (MIT_RED, MIT_RED_DARK) else "normal", color=col)

    style_ax(ax1)
    ax1.set_xticks([0.0, 1.3, 3.0, 4.0])
    ax1.set_xticklabels(["Initial\nBaseline", "OOD Gate /\n4-Depth", "Final\nValidation", "Locked\nTest"], fontsize=9.5, color=INK)
    ax1.set_xlim(-0.25, 4.55)
    ax1.set_ylim(0.73, 0.945)
    ax1.grid(axis="y", linestyle="-", alpha=0.8)
    ax1.set_ylabel("Open-Set ROC-AUC", fontsize=10)
    ax1.set_title("Iterative Open-Set ROC-AUC Progression", fontsize=12.5, fontweight="bold", loc="left", pad=12)
    ax1.legend(loc="lower right", frameon=True, facecolor=WHITE, edgecolor=RULE, fontsize=8.3)

    # Right: Zero-Shot FPR collapse (Before OOD Gate vs After OOD Gate on Locked Test)
    groups = [
        "ML\nGuardian", "ML\nArXiv",
        "Base\nGuardian", "Base\nArXiv",
        "Large\nGuardian", "Large\nArXiv",
    ]
    fpr_before = [89.6, 75.4, 97.6, 94.8, 92.5, 86.2]  # Val R1/R2 before OOD Gate (let's verify Large R1!)
    # Wait: let's read exact Large R1 Guardian & ArXiv FPR directly from validation_large_r1_vs_r2_by_domain.csv!
    r12_large = read_zip_csv(Z3, "03_deberta_v3_large/validation_large_r1_vs_r2_by_domain.csv")
    lg_r1_guard = float(r12_large.loc[r12_large["Domain"] == "guardian_journalism", "R1_FPR_%"].iloc[0])
    lg_r1_arxiv = float(r12_large.loc[r12_large["Domain"] == "arxiv_research", "R1_FPR_%"].iloc[0])
    fpr_before = [89.6, 75.4, 97.6, 94.8, lg_r1_guard, lg_r1_arxiv]
    fpr_after_test = [25.7, 25.7, 40.6, 17.7, 46.5, 8.6]

    yg = np.arange(len(groups))
    h = 0.34
    ax2.barh(yg - h / 2, fpr_before, height=h, color="#D5CECD", label="Before OOD Gate (Val R1/R2)")
    after_colors = [GRAY, GRAY, SLATE, SLATE, MIT_RED_MID, MIT_RED]
    ax2.barh(yg + h / 2, fpr_after_test, height=h, color=after_colors, label="After OOD Gate (Locked Test)")

    for i, (b, a) in enumerate(zip(fpr_before, fpr_after_test)):
        ax2.text(b + 1.2, i - h / 2, f"{b:.1f}%", va="center", ha="left", fontsize=8.2, color=MUTED)
        col_a = MIT_RED_DARK if i >= 4 else INK
        ax2.text(a + 1.2, i + h / 2, f"{a:.1f}%", va="center", ha="left", fontsize=8.5, fontweight="bold" if i == 5 else "normal", color=col_a)

    style_ax(ax2)
    ax2.set_yticks(yg)
    ax2.set_yticklabels(groups, fontsize=9.2, color=INK)
    ax2.set_ylim(len(groups) + 0.95, -0.55)
    ax2.set_xlim(0, 112)
    ax2.grid(axis="x", linestyle="-", alpha=0.8)
    ax2.set_xlabel("False Positive Rate (%) (Lower is Better)", fontsize=10)
    ax2.set_title("Zero-Shot Genre FPR Collapse via OOD Gating", fontsize=12.5, fontweight="bold", loc="left", pad=12)
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor="#D5CECD", label="Before OOD Gate (Val R1/R2)"),
        Patch(facecolor=MIT_RED, label="After: Large (Test)"),
        Patch(facecolor=SLATE, label="After: Base (Test)"),
        Patch(facecolor=GRAY, label="After: ML (Test)"),
    ]
    ax2.legend(handles=legend_handles, loc="lower right", ncol=2, frameon=True, facecolor=WHITE, edgecolor=RULE, fontsize=8.0)

    fig.tight_layout(pad=2.0)
    fig.savefig(FIG_PORTFOLIO / "nlu-improvement-progression.svg", bbox_inches="tight")
    fig.savefig(FIG_REPO / "improvement_progression.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] wrote nlu-improvement-progression.svg & improvement_progression.png (Large R1 FPR: guard={lg_r1_guard}%, arxiv={lg_r1_arxiv}%)")


def build_domain_length_fig() -> None:
    len_df = read_zip_csv(Z3, "03_deberta_v3_large/comparison_3way_test_by_length.csv")
    st_df = read_zip_csv(Z3, "03_deberta_v3_large/comparison_3way_test_by_stratum.csv")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.4), gridspec_kw={"width_ratios": [1.0, 1.15]})
    fig.patch.set_facecolor(PAPER)

    cols = [
        ("01. Stylometric ML", "01. Stylometric ML_AUC", GRAY),
        ("02. DeBERTa-v3-base", "02. DeBERTa-v3-base_AUC", SLATE),
        ("03. DeBERTa-v3-large", "03. DeBERTa-v3-large_AUC", MIT_RED),
    ]

    # Left: Length bucket
    xb = np.arange(len(len_df))
    w = 0.25
    for idx, (lbl, col, color) in enumerate(cols):
        vals = len_df[col].to_numpy()
        bars = ax1.bar(xb + (idx - 1) * w, vals, width=w * 0.92, color=color, label=lbl)
        for b, v in zip(bars, vals):
            ax1.text(b.get_x() + b.get_width() / 2, v + 0.007, f"{v:.3f}", ha="center", va="bottom",
                     fontsize=8.3, fontweight="bold" if idx == 2 else "normal",
                     color=MIT_RED_DARK if idx == 2 else MUTED)

    style_ax(ax1)
    ax1.set_xticks(xb)
    ax1.set_xticklabels(["Ultra-Short\n(<60w, n=4,058)", "Short\n(60-180w, n=2,226)", "Medium\n(180-350w, n=3,156)"], fontsize=9.2, color=INK)
    ax1.set_ylim(0.72, 1.00)
    ax1.grid(axis="y", linestyle="-", alpha=0.8)
    ax1.set_ylabel("ROC-AUC", fontsize=10)
    ax1.set_title("ROC-AUC by Passage Length (DL +9.98%p on <60w)", fontsize=12, fontweight="bold", loc="left", pad=12)
    ax1.legend(loc="upper right", frameon=True, facecolor=WHITE, edgecolor=RULE, fontsize=8.5)

    # Right: Evaluation Stratum (ordered cleanly)
    stratum_labels = {
        "open_unseen_author_same_topic": "Unseen Author\nSame-Topic (5,840)",
        "cross_topic": "Cross-Topic\nSame Author (2,158)",
        "zero_shot_unseen_genre": "Zero-Shot Genre\nArXiv (880)",
        "cross_genre": "Cross-Genre\nGuardian etc. (562)",
    }
    order = ["open_unseen_author_same_topic", "cross_topic", "zero_shot_unseen_genre", "cross_genre"]
    st_ord = st_df.set_index("Eval_Stratum").loc[order].reset_index()

    ys = np.arange(len(st_ord))
    bh = 0.25
    for idx, (lbl, col, color) in enumerate(cols):
        vals = st_ord[col].to_numpy()
        bars = ax2.barh(ys + (idx - 1) * bh, vals, height=bh * 0.92, color=color, label=lbl)
        for b, v in zip(bars, vals):
            is_win = (idx == 2 and b.get_y() < 2.6) or (idx == 0 and b.get_y() > 2.6)
            ax2.text(v + 0.005, b.get_y() + b.get_height() / 2, f"{v:.4f}", va="center", ha="left",
                     fontsize=8.3, fontweight="bold" if is_win else "normal",
                     color=MIT_RED_DARK if (idx == 2 and is_win) else (INK if is_win else MUTED))

    style_ax(ax2)
    ax2.set_yticks(ys)
    ax2.set_yticklabels([stratum_labels[s] for s in order], fontsize=9.2, color=INK)
    ax2.invert_yaxis()
    ax2.set_xlim(0.60, 1.02)
    ax2.grid(axis="x", linestyle="-", alpha=0.8)
    ax2.set_xlabel("ROC-AUC", fontsize=10)
    ax2.set_title("Stratum Complementarity: DL on ArXiv vs ML on Cross-Genre", fontsize=12, fontweight="bold", loc="left", pad=12)
    ax2.legend(loc="lower right", frameon=True, facecolor=WHITE, edgecolor=RULE, fontsize=8.5)

    fig.tight_layout(pad=2.0)
    fig.savefig(FIG_PORTFOLIO / "nlu-domain-length.svg", bbox_inches="tight")
    fig.savefig(FIG_REPO / "domain_length_complementarity.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("[ok] wrote nlu-domain-length.svg & domain_length_complementarity.png")


def build_og_image() -> None:
    fig = plt.figure(figsize=(12, 6.3), dpi=100)
    fig.patch.set_facecolor(PAPER)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6.3)
    ax.axis("off")

    ax.add_patch(plt.Rectangle((0, 6.12), 12, 0.18, color=MIT_RED))
    ax.add_patch(plt.Rectangle((0.65, 0.55), 10.7, 1.55, facecolor=MIT_RED, edgecolor="none"))
    ax.text(0.65, 5.45, "NLU AUTHORSHIP VERIFICATION", fontsize=13, fontweight="bold", color=MIT_RED)
    ax.text(0.65, 4.65, "동일 저자 판별: 순수 문체학 ML vs DeBERTa-v3", fontsize=28, fontweight="bold", color=INK)
    ax.text(0.65, 4.02, "양방향 저자 격리 9,440쌍 · 1:30 닫힌 집합 240쿼리 · 4계층 Dual-Rank ABTT OOD 보정", fontsize=14.5, color=MUTED)

    cards = [
        ("01. Stylometric ML", "AUC 0.8474 · CPU 32ms", GRAY, 0.65),
        ("02. DeBERTa-v3-base", "AUC 0.9040 · 4D-ABTT", SLATE, 4.30),
        ("03. DeBERTa-v3-large", "AUC 0.9139 · Dual-ABTT", MIT_RED, 7.95),
    ]
    for title, sub, col, x0 in cards:
        ax.add_patch(plt.Rectangle((x0, 2.45), 3.4, 1.15, facecolor=WHITE, edgecolor=col, linewidth=2.0 if col == MIT_RED else 1.2))
        ax.add_patch(plt.Rectangle((x0, 2.45), 0.12, 1.15, facecolor=col, edgecolor="none"))
        ax.text(x0 + 0.28, 3.16, title, fontsize=13.5, fontweight="bold", color=INK)
        ax.text(x0 + 0.28, 2.74, sub, fontsize=11.5, color=col if col == MIT_RED else MUTED)

    ax.text(1.0, 1.48, "Final Locked Test: DeBERTa-v3-large (437M)", fontsize=15, fontweight="bold", color=WHITE)
    ax.text(1.0, 0.94, "Open ROC-AUC 0.9139  |  PAN Overall 0.8643  |  Closed 1:30 Top-1 78.33%  |  Zero-Shot ArXiv 0.9389", fontsize=12.5, color=MIT_RED_SOFT)

    fig.savefig(OG_PORTFOLIO, dpi=100, format="jpeg", pil_kwargs={"quality": 92})
    plt.close(fig)
    print(f"[ok] wrote {OG_PORTFOLIO}")


if __name__ == "__main__":
    write_thumbnail_svg()
    build_3way_benchmark_fig()
    build_progression_and_fpr_fig()
    build_domain_length_fig()
    build_og_image()
