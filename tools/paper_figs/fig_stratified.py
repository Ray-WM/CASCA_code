"""Vector redraw of the reliability-stratified MPJPE figure (paper Fig. 4).

Data: reliability-stratified RICH numbers of the paper (all five
controlled variants). The heavy stratum is drawn but its small n is
flagged in the paper caption, not here.

Usage: python tools/paper_figs/fig_stratified.py
Out:   paper/pic/Fig_stratified.pdf
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))
from paper_style import MODEL_ORDER, W_COL, setup, style_of

import numpy as np
import matplotlib.pyplot as plt

# RICH stratified MPJPE (mm) — rows follow MODEL_ORDER
DATA = {
    "control": [68.7, 117.0, 143.8],
    "control_l": [67.2, 114.1, 140.8],
    "fuze": [72.5, 131.6, 162.6],
    "crossatt": [70.8, 131.8, 180.5],
    "casca": [72.6, 121.9, 146.9],
}
STRATA = ["Clean\n($n$=103,615)", "Mild\n($n$=2,669)", "Heavy\n($n$=162)"]


def main():
    setup()
    fig, ax = plt.subplots(figsize=(W_COL, 2.2))
    n_models = len(MODEL_ORDER)
    width = 0.16
    x = np.arange(len(STRATA))
    for i, tag in enumerate(MODEL_ORDER):
        st = style_of(tag)
        offs = (i - (n_models - 1) / 2) * width
        hatch = "//" if tag == "control_l" else None
        bars = ax.bar(
            x + offs, DATA[tag], width=width * 0.94, color=st["color"],
            label=st["label"], hatch=hatch, edgecolor="white", linewidth=0.4,
        )
        if tag in ("control", "casca"):  # selective direct labels only
            for b, v in zip(bars, DATA[tag]):
                ax.annotate(f"{v:.0f}", (b.get_x() + b.get_width() / 2, v),
                            xytext=(0, 1.5), textcoords="offset points",
                            ha="center", fontsize=6, color=st["color"])
    ax.set_xticks(x)
    ax.set_xticklabels(STRATA)
    ax.set_ylabel("MPJPE (mm)")
    ax.set_ylim(0, 185)  # bars encode length: keep the zero baseline
    ax.legend(ncols=2, loc="upper left")
    out = Path(__file__).parents[2] / "paper/pic/Fig_stratified.pdf"
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
