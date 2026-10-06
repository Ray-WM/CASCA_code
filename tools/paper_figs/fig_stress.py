"""Corruption stress-test line figure (the advantage figure).

Two panels: EMDB-2 and RICH WA-MPJPE vs corruption rate p, for
control / control+SG / control-L (if available) / CASCA.

Usage: python tools/paper_figs/fig_stress.py
Out:   Paper/TCSVT/pic/Fig_stress.pdf
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))
from paper_style import W_COL, setup, style_of

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).parents[2] / "results/e8_stress"
RATES = ["0.00", "0.05", "0.10", "0.20", "0.40"]
X = [0, 5, 10, 20, 40]


def arm(tag, arm_key="raw"):
    d = json.load(open(ROOT / f"{tag}.json"))
    return {ds: [d[r][arm_key][ds]["wa"] for r in RATES] for ds in ["EMDB_2", "RICH"]}


def main():
    setup()
    arms = [("control", arm("control"), style_of("control"), "GVHMR-control"),
            ("sg", arm("control", "sg"), style_of("sg"), "control + SG")]
    if (ROOT / "control_l.json").exists():
        arms.append(("control_l", arm("control_l"), style_of("control_l"), "GVHMR-control-L"))
    arms.append(("casca", arm("casca"), style_of("casca"), "CASCA (Ours)"))

    fig, axes = plt.subplots(1, 2, figsize=(W_COL, 2.0))
    for ax, ds, title in [(axes[0], "EMDB_2", "EMDB-2"), (axes[1], "RICH", "RICH")]:
        for key, data, st, label in arms:
            y = data[ds]
            kw = dict(st); kw["label"] = label
            lw = 1.8 if key == "casca" else 1.1
            ax.plot(X, y, marker="o", ms=2.5, lw=lw, **kw)
            ax.annotate(f"{y[-1]:.0f}", (X[-1], y[-1]), xytext=(3, 0),
                        textcoords="offset points", fontsize=6, color=kw["color"], va="center")
        ax.set_title(title, fontsize=8)
        ax.set_xlabel("corruption rate (%)")
        ax.set_xticks(X)
    axes[0].set_ylabel(r"WA-MPJPE (mm) $\downarrow$")
    axes[0].legend(fontsize=5.5, loc="upper left")
    out = Path(__file__).parents[2] / "Paper/TCSVT/pic/Fig_stress.pdf"
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
