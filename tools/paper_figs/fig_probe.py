"""Three-panel TNP probe figure (paper Fig_probe) from results/e3_probe.

Panels: constant offset / Gaussian noise / zero confidence — dMPJPE vs
offset from the corruption window, control vs CASCA, valid frames only.

Usage: python tools/paper_figs/fig_probe.py
Out:   Paper/TCSVT/pic/Fig_probe.pdf
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))
from paper_style import W_PAGE, setup, style_of

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).parents[2] / "results/e3_probe"
PANELS = [("offset", "Constant offset (gate-blind)"),
          ("gauss", "Gaussian noise (gate-blind)"),
          ("zeroconf", "Zero confidence (gate-visible)")]
MAXD = 20


def curve(tag, mode):
    clean = json.load(open(ROOT / f"{tag}_clean_emdb1.json"))
    noisy = json.load(open(ROOT / f"{tag}_{mode}_emdb1.json"))
    acc = defaultdict(list)
    for vid, nd in noisy.items():
        c = np.array(clean[vid]["mpjpe"]); n = np.array(nd["mpjpe"])
        v = np.array(nd["valid"], dtype=bool)
        center = (nd["noise_start"] + nd["noise_end"]) / 2.0
        for t in range(min(len(c), len(n))):
            d = int(round(t - center))
            if abs(d) <= MAXD and v[t]:
                acc[d].append(n[t] - c[t])
    ds = np.array(sorted(acc))
    return ds, np.array([np.mean(acc[d]) for d in ds])


def main():
    setup()
    fig, axes = plt.subplots(1, 3, figsize=(W_PAGE, 2.1), sharey=True)
    for ax, (mode, title) in zip(axes, PANELS):
        for tag, key in [("control", "control"), ("casca", "casca")]:
            ds, m = curve(tag, mode)
            st = style_of(key)
            ax.plot(ds, m, lw=1.5, **st)
        ax.axvspan(-2.5, 2.5, color="#888888", alpha=0.12, lw=0)
        ax.axhline(0, color="#888888", lw=0.6)
        ax.set_title(title, fontsize=8)
        ax.set_xlabel("offset from corruption window (frames)")
    axes[0].set_ylabel(r"$\Delta$MPJPE (mm)")
    axes[0].legend(loc="upper left")
    out = Path(__file__).parents[2] / "Paper/TCSVT/pic/Fig_probe.pdf"
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
