"""Compose the paper qualitative figure from results/e9_qualitative assets.

Per sequence (row): tight-cropped mesh overlays (control | CASCA) at the
most discriminative timestamp, top-down root trajectory, per-frame error
curve with d_t shading.

Usage: python tools/paper_figs/fig_qualitative.py
Out:   Paper/TCSVT/pic/Fig_qualitative.pdf
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))
from paper_style import W_PAGE, setup, style_of

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).parents[2] / "results/e9_qualitative"
SEQS = [
    ("test_Gym_010_dips2_cam_05", "RICH: dips (equipment occlusion)"),
    ("P4_37_outdoor_run_circle", "EMDB: running in a circle"),
    ("P8_69_outdoor_cartwheel", "EMDB: cartwheel"),
]


def crop_pair(seq_dir, t, margin=0.45):
    """Common tight crop around the union of both meshes."""
    ims = {}
    boxes = []
    for m in ["control", "casca"]:
        im = np.asarray(Image.open(seq_dir / f"overlay_{m}_f{t}.png").convert("RGB"))
        ims[m] = im
        nz = np.argwhere((im < 245).any(-1))
        if len(nz):
            (y0, x0), (y1, x1) = nz.min(0), nz.max(0)
            boxes.append((y0, x0, y1, x1))
    y0 = min(b[0] for b in boxes); x0 = min(b[1] for b in boxes)
    y1 = max(b[2] for b in boxes); x1 = max(b[3] for b in boxes)
    h, w = y1 - y0, x1 - x0
    my, mx = int(h * margin), int(w * margin)
    H, W = ims["control"].shape[:2]
    y0, y1 = max(0, y0 - my), min(H, y1 + my)
    x0, x1 = max(0, x0 - mx), min(W, x1 + mx)
    # make near-square-ish
    return {m: ims[m][y0:y1, x0:x1] for m in ims}


def main():
    setup()
    fig, axes = plt.subplots(3, 4, figsize=(W_PAGE, 5.6),
                             gridspec_kw={"width_ratios": [1, 1, 1.15, 1.7]})
    for r, (seq, title) in enumerate(SEQS):
        sd = ROOT / seq
        picks = json.load(open(sd / "picks.json"))
        t = picks["picks"][0]
        crops = crop_pair(sd, t)
        fm = picks.get("frame_mpjpe", {})
        for c, (m, lab) in enumerate([("control", "GVHMR-control"), ("casca", "CASCA")]):
            ax = axes[r, c]
            ax.imshow(crops[m]); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            col = style_of(m if m == "casca" else "control")["color"]
            for s in ax.spines.values():
                s.set_visible(True); s.set_color(col); s.set_linewidth(1.6)
            if m in fm:
                ax.set_xlabel(f"{fm[m][0]:.0f} mm", fontsize=7.5, color=col, labelpad=2)
            if r == 0:
                ax.set_title(lab, fontsize=8, color=col)

        tr = np.load(sd / "traj.npz")

        def align(p, g, k=30):
            p = p - p[0:1] + g[0:1]
            k = min(k, len(p) - 1, len(g) - 1)
            va, vb = p[k] - p[0], g[k] - g[0]
            a1 = np.arctan2(va[2], va[0]); a2 = np.arctan2(vb[2], vb[0])
            th = a2 - a1
            R = np.array([[np.cos(th), 0, np.sin(th)], [0, 1, 0],
                          [-np.sin(th), 0, np.cos(th)]])
            return (p - g[0:1]) @ R.T + g[0:1]

        ax = axes[r, 2]
        gt = tr["gt"]
        for key, tag in [("gt", None), ("control", "control"), ("casca", "casca")]:
            arr = tr[key] if key == "gt" else align(tr[key], gt)
            xy = arr[:, [0, 2]]
            if tag is None:
                ax.plot(xy[:, 0], xy[:, 1], color="#C4C4C4", lw=2.4, label="GT",
                        solid_capstyle="round")
            else:
                st = style_of(tag)
                ax.plot(xy[:, 0], xy[:, 1], lw=1.0, **st)
                ax.plot(xy[0, 0], xy[0, 1], "o", ms=3, color=st["color"])
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        # 1 m scale bar (lower left)
        x0, x1 = ax.get_xlim(); y0, _ = ax.get_ylim()
        ax.plot([x0 + 0.06*(x1-x0), x0 + 0.06*(x1-x0) + 1.0], [y0]*2, color="#666666", lw=1.4)
        ax.text(x0 + 0.06*(x1-x0) + 0.5, y0, "1 m", fontsize=6, ha="center", va="bottom", color="#666666")
        if r == 0:
            ax.set_title("top-down trajectory", fontsize=8)

        cv = np.load(sd / "curves.npz")
        ax = axes[r, 3]
        v = cv["valid"].astype(bool)[: len(cv["mpjpe_control"])]
        f = np.arange(len(cv["mpjpe_control"]))
        for tag in ["control", "casca"]:
            y = cv[f"mpjpe_{tag}"].copy(); y[~v] = np.nan
            ax.plot(f, y, lw=0.9, **style_of(tag))
        ax2 = ax.twinx()
        ax2.fill_between(f, 0, np.nan_to_num(cv["d_t"][: len(f)]), color="#E69F00",
                         alpha=0.25, lw=0)
        ax2.set_ylim(0.50, 0.60); ax2.set_yticks([]); ax2.grid(False)
        ax.axvline(t, color="#555555", lw=0.7, ls=":")
        ax.set_ylabel("MPJPE (mm)", fontsize=7)
        ax.set_ylabel(title, fontsize=8)  # row label via left axis of last col? keep simple
        ax.set_ylabel("MPJPE (mm)", fontsize=7)
        ax.text(0.02, 0.93, title, transform=ax.transAxes, fontsize=7.5, va="top")
        if r == 0:
            ax.set_title(r"per-frame error ($d_t$ shaded)", fontsize=8)
        if r == 2:
            ax.set_xlabel("frame")
    h1, l1 = axes[0, 2].get_legend_handles_labels()
    h2, l2 = axes[0, 3].get_legend_handles_labels()
    seen, H, L = set(), [], []
    for h, l in zip(h1 + h2, l1 + l2):
        if l not in seen:
            seen.add(l); H.append(h); L.append(l)
    fig.legend(H, L, fontsize=7, ncols=3, loc="lower center",
               bbox_to_anchor=(0.5, -0.015), frameon=False)
    out = Path(__file__).parents[2] / "Paper/TCSVT/pic/Fig_qualitative.pdf"
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
