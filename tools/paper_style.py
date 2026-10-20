"""Shared matplotlib style for the paper figures.

One entity = one color across every figure (color follows the entity):
  CASCA     : blue   #0072B2  (ours — the standout)
  Fuze      : orange #D55E00
  CrossAtt  : green  #009E73
  control   : gray   #555555, solid   (GVHMR-control baseline)
  control_l : gray   #999999, dashed  (parameter-matched control)
Colored trio validated CVD-safe; the two controls are deliberately neutral
(reference role) and are always disambiguated by linestyle + legend label.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODEL_ORDER = ["control", "control_l", "fuze", "crossatt", "casca"]

MODEL_STYLE = {
    "control": dict(color="#555555", ls="-", label="GVHMR-control"),
    "control_l": dict(color="#999999", ls="--", label="GVHMR-control-L"),
    "fuze": dict(color="#D55E00", ls="-", label="Fuze"),
    "crossatt": dict(color="#009E73", ls="-", label="CrossAtt"),
    "casca": dict(color="#0072B2", ls="-", label="CASCA (Ours)"),
    # smoothing baselines (used only in E1-related figures)
    "sg": dict(color="#CC79A7", ls=":", label="+ Savitzky-Golay"),
    "oneeuro": dict(color="#E69F00", ls=":", label="+ OneEuro"),
    "smoothnet": dict(color="#56B4E9", ls=":", label="+ SmoothNet"),
}

# Two-column journal column and page widths (inches)
W_COL = 3.5
W_PAGE = 7.16


def setup(font_scale=1.0):
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,  # embed TrueType fonts for publisher PDF checks
            "ps.fonttype": 42,
            "font.family": "serif",
            "font.size": 8 * font_scale,
            "axes.titlesize": 8.5 * font_scale,
            "axes.labelsize": 8 * font_scale,
            "xtick.labelsize": 7 * font_scale,
            "ytick.labelsize": 7 * font_scale,
            "legend.fontsize": 7 * font_scale,
            "axes.linewidth": 0.6,
            "grid.linewidth": 0.4,
            "grid.alpha": 0.25,
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "lines.linewidth": 1.4,
            "legend.frameon": False,
            "figure.constrained_layout.use": True,
        }
    )


def style_of(tag):
    return MODEL_STYLE[tag.lower()]
