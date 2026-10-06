"""E7 attention-decay figure: mean attention weight vs temporal distance
(log scale) for the trained control encoder, with exponential fit.

Usage: python tools/paper_figs/fig_attn_decay.py
Out:   Paper/TCSVT/pic/Fig_attn_decay.pdf
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))
from paper_style import W_COL, setup, style_of

import numpy as np
import matplotlib.pyplot as plt


def main():
    setup()
    d = json.load(open(Path(__file__).parents[2] / "results/e7_attention/control/decay.json"))
    mean = np.array(d["mean_curve"], dtype=float)
    fig, ax = plt.subplots(figsize=(W_COL, 1.8))
    for k, v in d["layers"].items():
        ax.plot(np.arange(len(v)), v, color="#0072B2", alpha=0.15, lw=0.6)
    ax.plot(np.arange(len(mean)), mean, color="#0072B2", lw=1.6, label="layer mean")
    beta = d["beta_hat_per_frame"]
    x = np.arange(1, len(mean))
    fit = mean[1] * beta ** (x - 1)
    ax.plot(x, fit, color="#555555", ls="--", lw=1.0,
            label=rf"$\hat\beta={beta:.3f}$ / frame")
    for dist, frac in [(d["dist_to_1_10"], "1/10"), (d["dist_to_1_100"], "1/100")]:
        ax.axvline(dist, color="#999999", lw=0.6, ls=":")
        ax.annotate(f"{frac}@{dist}f", (dist, mean[1] * 0.6), fontsize=6,
                    color="#666666", rotation=90, va="top", ha="right")
    ax.set_yscale("log")
    ax.set_xlim(0, 120)
    ax.set_xlabel(r"temporal distance $|t-s|$ (frames)")
    ax.set_ylabel("mean attention")
    ax.legend(fontsize=6.5)
    out = Path(__file__).parents[2] / "Paper/TCSVT/pic/Fig_attn_decay.pdf"
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
