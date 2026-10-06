"""E4-R dose-response analysis (local; consumes results/e4r/<tag>/<dataset>.npz).

Per dataset:
  1. Dose-response curves: per-frame MPJPE vs. observation degradation
     (n_lowconf bins and g_norm decile bins), one line per model, with
     per-sequence bootstrap CIs.  -> results/e4r/analysis/<ds>_dose_<x>.pdf
  2. Linear fits MPJPE = a + b*g per model with bootstrap CIs on (a, b).
  3. Paired stats for CASCA - control: Spearman rho(delta, g), top-decile
     contribution share, stratified means, per-frame vs per-sequence
     averaging (weighted-reconstruction check).
  4. Gate-information stats (CASCA only): corr(d_t, g), d_t by stratum.
Headline numbers -> results/e4r/analysis/analysis_numbers.json, matching
the \\TODO{E4-R}/\\TODO{E6} markers in the TCSVT tex.

Bootstrap resamples SEQUENCES (not frames) using per-sequence sufficient
statistics, so 10k iterations stay fast even on 100k-frame datasets.

Usage:
  python tools/e4r_analysis.py [--root results/e4r] [--boot 10000]
                               [--tags control,control_l,fuze,crossatt,casca]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from paper_style import MODEL_ORDER, W_COL, setup, style_of  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

DATASETS = ["RICH", "EMDB_1", "EMDB_2", "3DPW"]
NLOW_EDGES = np.array([0, 1, 2, 3, 4, 5, 6], dtype=float)  # last bin open (6+)
RNG = np.random.default_rng(0)


def load(root, tag, ds):
    fn = Path(root) / tag / f"{ds}.npz"
    if not fn.exists():
        return None
    d = dict(np.load(fn, allow_pickle=False))
    m = d["valid"].astype(bool) & np.isfinite(d["mpjpe"]) & np.isfinite(d["g_norm"])
    return {k: v[m] for k, v in d.items()}


def seq_groups(vids):
    uniq, inv = np.unique(vids, return_inverse=True)
    return uniq, inv


# ---------- per-sequence sufficient statistics ----------

def fit_stats(g, y, inv, n_seq):
    """Per-seq sums for OLS y = a + b*g."""
    S = np.zeros((n_seq, 5))
    np.add.at(S, inv, np.stack([np.ones_like(g), g, y, g * g, g * y], axis=1))
    return S  # cols: n, Sx, Sy, Sxx, Sxy


def fit_from_sums(S):
    n, Sx, Sy, Sxx, Sxy = S.sum(0)
    denom = n * Sxx - Sx * Sx
    b = (n * Sxy - Sx * Sy) / denom if denom else np.nan
    a = (Sy - b * Sx) / n
    return a, b


def bin_stats(x, y, inv, n_seq, edges):
    """Per-seq (sum, count) of y within each x-bin (last bin open)."""
    n_bin = len(edges)
    idx = np.searchsorted(edges, x, side="right") - 1
    idx = np.clip(idx, 0, n_bin - 1)
    sums = np.zeros((n_seq, n_bin))
    cnts = np.zeros((n_seq, n_bin))
    np.add.at(sums, (inv, idx), y)
    np.add.at(cnts, (inv, idx), 1.0)
    return sums, cnts


def boot_ci(per_seq_stats, reducer, n_boot, n_seq):
    """Generic sequence bootstrap: reducer(list of resampled row-sums) -> value(s)."""
    out = []
    for _ in range(n_boot):
        rows = RNG.integers(0, n_seq, n_seq)
        out.append(reducer([s[rows] for s in per_seq_stats]))
    out = np.asarray(out, dtype=float)
    return np.nanpercentile(out, [2.5, 97.5], axis=0)


def spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = np.sqrt((rx**2).sum() * (ry**2).sum())
    return float((rx * ry).sum() / denom) if denom > 0 else float("nan")


# ---------- main analysis ----------

def analyze_dataset(root, ds, tags, n_boot, out_dir):
    data = {t: d for t in tags if (d := load(root, t, ds)) is not None}
    if not data:
        return None
    print(f"\n=== {ds}: {', '.join(data)} ===")
    res = {"models": {}}

    ref = data.get("control") or next(iter(data.values()))
    dec_edges = np.unique(np.percentile(ref["g_norm"], np.arange(0, 100, 10)))

    curves = {}  # tag -> (xkey -> (means, lo, hi))
    for t, d in data.items():
        g, y, nlow = d["g_norm"], d["mpjpe"], d["n_lowconf"]
        uniq, inv = seq_groups(d["vid"])
        n_seq = len(uniq)

        # --- linear fit + CI ---
        S = fit_stats(g, y, inv, n_seq)
        a, b = fit_from_sums(S)
        ci = boot_ci([S], lambda rs: np.array(fit_from_sums(rs[0])), n_boot, n_seq)
        a_ci, b_ci = ci[:, 0], ci[:, 1]

        # --- strata (binary thresholds, cross-check vs. dose-response) ---
        mild_m = (nlow >= 3) & (nlow < 6)
        heavy_m = nlow >= 6
        strata = {
            "clean": float(y[nlow < 3].mean()),
            "mild": float(y[mild_m].mean()) if mild_m.any() else None,
            "heavy": float(y[heavy_m].mean()) if heavy_m.any() else None,
            "n_clean": int((nlow < 3).sum()),
            "n_mild": int(mild_m.sum()),
            "n_heavy": int(heavy_m.sum()),
            "n_seqs_mild": int(len(np.unique(d["vid"][mild_m]))) if mild_m.any() else 0,
            "n_seqs_heavy": int(len(np.unique(d["vid"][heavy_m]))) if heavy_m.any() else 0,
        }
        per_seq_mean = float(np.mean([y[inv == i].mean() for i in range(n_seq)]))
        res["models"][t] = {
            "n_frames": int(len(y)),
            "n_seqs": n_seq,
            "overall_perframe": float(y.mean()),
            "overall_perseq": per_seq_mean,
            "intercept_a": float(a),
            "slope_b": float(b),
            "a_ci": a_ci.tolist(),
            "b_ci": b_ci.tolist(),
            "strata": strata,
        }
        print(
            f"  {t:10s} a={a:7.2f} [{a_ci[0]:.2f},{a_ci[1]:.2f}]"
            f"  b={b:7.3f} [{b_ci[0]:.3f},{b_ci[1]:.3f}]"
            f"  overall {y.mean():.1f} (per-seq {per_seq_mean:.1f})"
        )

        # --- binned curves + CI (per-seq sums bootstrap) ---
        curves[t] = {}
        for xkey, edges in [("g_norm", dec_edges), ("n_lowconf", NLOW_EDGES)]:
            sums, cnts = bin_stats(d[xkey], y, inv, n_seq, edges)
            means = np.where(cnts.sum(0) > 0, sums.sum(0) / np.maximum(cnts.sum(0), 1), np.nan)
            lo, hi = boot_ci(
                [sums, cnts],
                lambda rs: rs[0].sum(0) / np.maximum(rs[1].sum(0), 1e-9),
                min(n_boot, 2000),
                n_seq,
            )
            small = cnts.sum(0) < 30
            lo, hi = lo.copy(), hi.copy()
            curves[t][xkey] = (means, lo, hi, cnts.sum(0).astype(int), small)

    if "casca" in res["models"] and "control" in res["models"]:
        b0 = res["models"]["control"]["slope_b"]
        b1 = res["models"]["casca"]["slope_b"]
        res["slope_reduction_pct"] = float(100 * (b0 - b1) / abs(b0)) if b0 else None

    # --- paired CASCA - control ---
    if "casca" in data and "control" in data:
        dc, db = data["casca"], data["control"]
        key_c = np.char.add(dc["vid"].astype(str), np.char.mod("#%d", dc["frame"].astype(int)))
        key_b = np.char.add(db["vid"].astype(str), np.char.mod("#%d", db["frame"].astype(int)))
        _, ic, ib = np.intersect1d(key_c, key_b, return_indices=True)
        delta = dc["mpjpe"][ic] - db["mpjpe"][ib]  # negative = CASCA better
        g = dc["g_norm"][ic]
        thr = np.percentile(g, 90)
        total_impr = -delta.sum()
        res["paired"] = {
            "n": int(len(delta)),
            "mean_delta": float(delta.mean()),
            "rho_delta_vs_g": spearman(g, delta),
            "top10pct_g_contribution_pct": float(-delta[g >= thr].sum() / total_impr * 100)
            if total_impr != 0
            else None,
        }
        if np.isfinite(dc["d_t"]).any():
            nlow_c = dc["n_lowconf"]
            mild_m = (nlow_c >= 3) & (nlow_c < 6)
            res["gate"] = {
                "rho_dt_vs_g": spearman(dc["g_norm"], dc["d_t"]),
                "rho_dt_vs_nlow": spearman(nlow_c, dc["d_t"]),
                "dt_clean": float(dc["d_t"][nlow_c < 3].mean()),
                "dt_mild": float(dc["d_t"][mild_m].mean()) if mild_m.any() else None,
                "dt_heavy": float(dc["d_t"][nlow_c >= 6].mean()) if (nlow_c >= 6).any() else None,
            }

    # --- figures ---
    for xkey, edges, xlabel, fname in [
        ("g_norm", dec_edges, "detection reprojection error, decile", f"{ds}_dose_g"),
        ("n_lowconf", NLOW_EDGES, "# low-confidence joints", f"{ds}_dose_nlow"),
    ]:
        fig, ax = plt.subplots(figsize=(W_COL, 2.3))
        for t in [m for m in MODEL_ORDER if m in curves]:
            means, lo, hi, ns, small = curves[t][xkey]
            x = np.arange(len(means))
            st = style_of(t)
            ax.plot(x, means, marker="o", ms=2.5, **st)
            ax.fill_between(x, lo, hi, color=st["color"], alpha=0.12, lw=0)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("per-frame MPJPE (mm)")
        if xkey == "n_lowconf":
            ax.set_xticks(np.arange(len(NLOW_EDGES)))
            ax.set_xticklabels([f"{int(e)}" for e in NLOW_EDGES[:-1]] + ["6+"])
        else:
            ax.set_xticks(np.arange(len(edges)))
            ax.set_xticklabels([f"{10 * i}" for i in range(len(edges))])
        ax.legend(ncols=2)
        fig.savefig(out_dir / f"{fname}.pdf")
        plt.close(fig)
        print(f"  wrote {out_dir / (fname + '.pdf')}")

    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/e4r")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--tags", default="control,control_l,fuze,crossatt,casca")
    args = ap.parse_args()

    setup()
    out_dir = Path(args.root) / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_res = {}
    for ds in DATASETS:
        r = analyze_dataset(args.root, ds, args.tags.split(","), args.boot, out_dir)
        if r:
            all_res[ds] = r

    with open(out_dir / "analysis_numbers.json", "w") as f:
        json.dump(all_res, f, indent=2)
    print(f"\nAll numbers -> {out_dir / 'analysis_numbers.json'}")


if __name__ == "__main__":
    main()
