"""P1 post-internal-review analyses (local, zero GPU).

Inputs : results/e4r/<model>/<DS>.npz  (per-frame export)
         results/e3_probe/<tag>_{clean,zeroconf,offset,gauss}_emdb1.json
Outputs: results/review_p1/p1_results.json (+ printed summary)

Analyses
  A. Incidence of confidently-wrong detections: P(g high | conf high), RICH/3DPW.
  B. Probe spillover bootstrap CIs (per-sequence, paired control-casca).
  C. Reliability-strata bootstrap CIs on RICH (cluster bootstrap by sequence).
  D. d_t per-frame distribution (overall + per stratum).
  E. Gate passthrough check: rho(mean_conf,g) vs rho(d_t,g) (+ per-seq rho).
"""

import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
E4R = ROOT / "results/e4r"
PROBE = ROOT / "results/e3_probe"
OUT = ROOT / "results/review_p1"
OUT.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(20260917)
N_BOOT = 10_000

out = {}


def load(model, ds):
    d = np.load(E4R / model / f"{ds}.npz", allow_pickle=True)
    m = d["valid"].astype(bool)
    return {k: d[k][m] for k in d.files}


# ---------------- A. incidence ----------------
print("=" * 70)
print("A. Incidence of confidently-wrong detections (control export)")
# g_norm units: pixels at a 256-px bounding-box reference
# (g_norm = g_pix / bbx_size * 256). The probe corruption of 0.3 bbox units
# corresponds to g_norm ~ 76.8. Dose-response slope ~4 mm MPJPE per unit.
THRS = [5.0, 10.0, 20.0, 40.0, 76.8]
inc = {}
for ds in ["RICH", "3DPW"]:
    d = load("control", ds)
    g = d["g_norm"]
    conf_hi = (d["mean_conf"] >= 0.8) & (d["n_lowconf"] == 0)
    q = {f"q{p}": float(np.percentile(g, p)) for p in [50, 90, 95, 99, 99.9]}
    row = {"n_frames": int(g.size), "frac_conf_hi": float(conf_hi.mean()), "g_norm_quantiles": q}
    for thr in THRS:
        g_hi = g >= thr
        both = conf_hi & g_hi
        row[f"P(g>={thr})"] = float(g_hi.mean())
        row[f"P(g>={thr}|conf_hi)"] = float(both.sum() / max(conf_hi.sum(), 1))
        row[f"P(g>={thr} & conf_hi)"] = float(both.mean())
    inc[ds] = row
    print(f"  {ds}: n={row['n_frames']}, conf_hi frac={row['frac_conf_hi']:.3f}, "
          f"g_norm quantiles={ {k: round(v,2) for k,v in q.items()} }")
    for thr in THRS:
        print(f"    P(g>={thr:g}|conf_hi) = {row[f'P(g>={thr}|conf_hi)']:.5f}  "
              f"joint = {row[f'P(g>={thr} & conf_hi)']:.5f}")
out["incidence"] = inc

# ---------------- B. probe CIs ----------------
print("=" * 70)
print("B. Probe spillover bootstrap CIs (offsets +1..+7)")


def spill_per_seq(tag, pert):
    """noise_end in the JSON is the LAST corrupted frame (inclusive);
    offsets +1..+7 are therefore noise_end+1 .. noise_end+7."""
    clean = json.load(open(PROBE / f"{tag}_clean_emdb1.json"))
    noisy = json.load(open(PROBE / f"{tag}_{pert}_emdb1.json"))
    per_seq, pooled = [], []
    for vid, nd in noisy.items():
        cd = clean[vid]
        mp_n, mp_c = np.array(nd["mpjpe"]), np.array(cd["mpjpe"])
        val = np.array(nd["valid"], bool)
        e = nd["noise_end"]
        idx = np.arange(e + 1, min(e + 8, len(mp_n)))
        idx = idx[val[idx]]
        if idx.size:
            deltas = mp_n[idx] - mp_c[idx]
            per_seq.append(float(deltas.mean()))
            pooled.extend(deltas.tolist())
    return np.array(per_seq), float(np.mean(pooled))


def boot_ci(x, stat=np.mean):
    bs = np.array([stat(RNG.choice(x, x.size, replace=True)) for _ in range(N_BOOT)])
    return float(stat(x)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


probe = {}
for pert in ["offset", "gauss", "zeroconf"]:
    sc, pooled_c = spill_per_seq("control", pert)
    sa, pooled_a = spill_per_seq("casca", pert)
    mc, lc, hc = boot_ci(sc)
    ma, la, ha = boot_ci(sa)
    diff = sc - sa  # paired (same sequences, same order)
    md, ld, hd = boot_ci(diff)
    probe[pert] = {
        "pooled_point": {"control": pooled_c, "casca": pooled_a},
        "control": {"mean": mc, "ci": [lc, hc]},
        "casca": {"mean": ma, "ci": [la, ha]},
        "paired_diff": {"mean": md, "ci": [ld, hd], "excl_zero": bool(ld > 0 or hd < 0)},
        "n_seq": int(sc.size),
    }
    print(f"  {pert:9s} pooled(paper) ctrl {pooled_c:+.2f} casca {pooled_a:+.2f} | "
          f"per-seq ctrl {mc:+.2f} [{lc:+.2f},{hc:+.2f}]  "
          f"casca {ma:+.2f} [{la:+.2f},{ha:+.2f}]  "
          f"paired diff {md:+.2f} [{ld:+.2f},{hd:+.2f}] "
          f"{'EXCLUDES 0' if probe[pert]['paired_diff']['excl_zero'] else 'includes 0'}")
out["probe_ci"] = probe

# ---------------- C. strata CIs ----------------
print("=" * 70)
print("C. RICH reliability-strata cluster-bootstrap CIs (resample sequences)")


def strata_ci(model):
    d = load(model, "RICH")
    vids = d["vid"]
    uvids = np.unique(vids)
    strat = np.where(d["n_lowconf"] >= 6, 2, np.where(d["n_lowconf"] >= 3, 1, 0))
    res = {}
    by_vid = {v: np.flatnonzero(vids == v) for v in uvids}
    for s, name in [(0, "clean"), (1, "mild"), (2, "heavy")]:
        point = float(d["mpjpe"][strat == s].mean())
        bs = []
        for _ in range(2000):
            pick = RNG.choice(uvids, uvids.size, replace=True)
            idx = np.concatenate([by_vid[v] for v in pick])
            m = strat[idx] == s
            if m.any():
                bs.append(d["mpjpe"][idx][m].mean())
        res[name] = {"mean": point, "ci": [float(np.percentile(bs, 2.5)),
                                           float(np.percentile(bs, 97.5))],
                     "n": int((strat == s).sum())}
    res["n_seq_heavy"] = int(len({v for v in uvids
                                  if (strat[by_vid[v]] == 2).any()}))
    return res


strata = {}
for model in ["control", "control_l", "casca"]:
    strata[model] = strata_ci(model)
    r = strata[model]
    print(f"  {model:10s} clean {r['clean']['mean']:.1f} {r['clean']['ci']}  "
          f"mild {r['mild']['mean']:.1f} {r['mild']['ci']}  "
          f"heavy {r['heavy']['mean']:.1f} {r['heavy']['ci']} "
          f"(heavy in {r['n_seq_heavy']} seqs)")
out["strata_ci"] = strata

# ---------------- D. d_t distribution ----------------
print("=" * 70)
print("D. d_t per-frame distribution (CASCA, RICH)")
d = load("casca", "RICH")
dt = d["d_t"]
strat = np.where(d["n_lowconf"] >= 6, 2, np.where(d["n_lowconf"] >= 3, 1, 0))
dist = {"quantiles": {f"q{p}": float(np.percentile(dt, p))
                      for p in [0.1, 1, 5, 25, 50, 75, 95, 99, 99.9]},
        "min": float(dt.min()), "max": float(dt.max()),
        "frac_outside_pm0.05": float((np.abs(dt - 0.5) > 0.05).mean()),
        "by_stratum": {name: {"mean": float(dt[strat == s].mean()),
                              "q99": float(np.percentile(dt[strat == s], 99))}
                       for s, name in [(0, "clean"), (1, "mild"), (2, "heavy")]}}
out["dt_distribution"] = dist
print(f"  range [{dist['min']:.3f}, {dist['max']:.3f}]  "
      f"q1={dist['quantiles']['q1']:.3f} q99={dist['quantiles']['q99']:.3f}  "
      f"frac |d-0.5|>0.05 = {dist['frac_outside_pm0.05']:.4f}")

# ---------------- E. passthrough ----------------
print("=" * 70)
print("E. Gate passthrough check (Spearman rho against g_norm)")
pt = {}
for ds in ["RICH", "3DPW"]:
    d = load("casca", ds)
    g = d["g_norm"]
    rows = {}
    for name, x in [("d_t", d["d_t"]), ("mean_conf", d["mean_conf"]),
                    ("n_lowconf", d["n_lowconf"])]:
        rho = spearmanr(x, g).statistic
        # per-sequence rho distribution
        seq_rhos = []
        for v in np.unique(d["vid"]):
            m = d["vid"] == v
            if m.sum() > 30:
                r = spearmanr(x[m], g[m]).statistic
                if np.isfinite(r):
                    seq_rhos.append(r)
        rows[name] = {"rho_frame": float(rho),
                      "rho_per_seq_median": float(np.median(seq_rhos)),
                      "rho_per_seq_iqr": [float(np.percentile(seq_rhos, 25)),
                                          float(np.percentile(seq_rhos, 75))]}
        print(f"  {ds} {name:10s} frame-rho={rho:+.3f}  "
              f"per-seq median={rows[name]['rho_per_seq_median']:+.3f} "
              f"IQR={rows[name]['rho_per_seq_iqr']}")
    pt[ds] = rows
out["passthrough"] = pt

json.dump(out, open(OUT / "p1_results.json", "w"), indent=1)
print("=" * 70)
print(f"saved -> {OUT / 'p1_results.json'}")
