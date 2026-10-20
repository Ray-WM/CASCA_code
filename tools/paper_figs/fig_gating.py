"""Vector redraw of the gating-behavior figure (paper Fig. 5).

Panels: (a) difficulty score d_t clean vs unreliable (RICH, EMDB-1);
        (b) cross-modal update norm clean vs unreliable (RICH, EMDB known).
Numbers from the MM manuscript Sec. 4.5 / rebuttal; the E4-R export will
add the d_t-vs-degradation curve as a third panel (placeholder flag below).

Usage: python tools/paper_figs/fig_gating.py
Out:   paper/pic/Fig_gating.pdf
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))
from paper_style import W_COL, setup

import numpy as np
import matplotlib.pyplot as plt

CLEAN_C, UNREL_C = "#999999", "#0072B2"

# (dataset, clean, unreliable)
# (dataset, clean, mild, heavy) — retrained model, E4-R export
DT3 = [("RICH", 0.526, 0.538, 0.549)]
RHO = [("RICH", 0.593), ("3DPW", 0.603)]


def grouped(ax, rows, ylabel, ylim):
    """Paired-dot (slope) panel — dots tolerate a zoomed axis, bars don't."""
    x = np.arange(len(rows))
    dx = 0.18
    for i, (name, clean, unrel) in enumerate(rows):
        ax.plot([i - dx, i + dx], [clean, unrel], color="#BBBBBB", lw=1.0, zorder=1)
        ax.plot(i - dx, clean, "o", ms=5, color=CLEAN_C, zorder=2,
                label="clean" if i == 0 else None)
        ax.plot(i + dx, unrel, "o", ms=5, color=UNREL_C, zorder=2,
                label="unreliable" if i == 0 else None)
        ax.annotate(f"{clean:.2f}", (i - dx, clean), xytext=(-2, -9),
                    textcoords="offset points", ha="right", fontsize=6, color="#555555")
        ax.annotate(f"{unrel:.2f}", (i + dx, unrel), xytext=(2, 3),
                    textcoords="offset points", ha="left", fontsize=6, color=UNREL_C)
    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in rows])
    ax.set_xlim(-0.6, len(rows) - 0.4)
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)


def main():
    setup()
    fig, axes = plt.subplots(1, 2, figsize=(W_COL, 1.9))
    ax = axes[0]
    for name, c, m, h in DT3:
        xs = [0, 1, 2]
        ax.plot(xs, [c, m, h], "-o", ms=4, color=UNREL_C)
        for x, v in zip(xs, [c, m, h]):
            ax.annotate(f"{v:.3f}", (x, v), xytext=(0, 5), ha="center",
                        textcoords="offset points", fontsize=6, color=UNREL_C)
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["clean", "mild", "heavy"])
    ax.set_ylabel("difficulty score $d_t$"); ax.set_ylim(0.515, 0.56)
    ax2 = axes[1]
    xs = range(len(RHO))
    ax2.bar(xs, [r for _, r in RHO], width=0.5, color=UNREL_C)
    for x, (n, r) in zip(xs, RHO):
        ax2.annotate(f"{r:.2f}", (x, r), xytext=(0, 2), ha="center",
                     textcoords="offset points", fontsize=7)
    ax2.set_xticks(list(xs)); ax2.set_xticklabels([n for n, _ in RHO])
    ax2.set_ylabel(r"Spearman $\rho(d_t, g_{\mathrm{GT}})$"); ax2.set_ylim(0, 0.75)
    out = Path(__file__).parents[2] / "paper/pic/Fig_gating.pdf"
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
