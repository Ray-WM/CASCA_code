"""E0: data-integrity checks gating the E4-R analysis (run on the REMOTE
machine, where inputs/ datasets exist). No GPU needed.

Resolves, with authoritative counts straight from the dataloaders:
  1. RICH heavy-stratum size: Appendix B.2 says n=12, C.3 says 0.2% (~213).
  2. EMDB frame counts vs. official durations (EMDB-1 13.5 min, EMDB-2
     24.0 min at 30 fps) — Appendix Table 2 lists 9,324 / 53,383.
  3. Effect of the EMDB/3DPW valid masks on the strata (are the heaviest
     frames excluded by construction?).
  4. Reliability distribution per dataset, on all frames AND valid-only.

Usage:  python tools/e0_integrity_check.py [--datasets rich,emdb1,emdb2,3dpw]
Output: results/e0_integrity/report.json + console summary.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

# Appendix Table 2 numbers from the MM submission (to compare against)
APPENDIX_TABLE2 = {
    "RICH": dict(frames=106446, clean=0.973, mild=0.025, heavy=0.002),
    "EMDB_1": dict(frames=9324, clean=0.992, mild=0.006, heavy=0.002),
    "EMDB_2": dict(frames=53383, clean=0.985, mild=0.010, heavy=0.006),
}
OFFICIAL_MINUTES = {"EMDB_1": 13.5, "EMDB_2": 24.0}  # at 30 fps


def strata_counts(conf, thr=0.5):
    nlow = (conf < thr).sum(-1)  # (L,)
    return {
        "clean": int((nlow < 3).sum()),
        "mild": int(((nlow >= 3) & (nlow < 6)).sum()),
        "heavy": int((nlow >= 6).sum()),
    }


def merge(a, b):
    return {k: a.get(k, 0) + b.get(k, 0) for k in set(a) | set(b)}


def check_dataset(name, ds):
    out = {
        "n_seqs": len(ds),
        "frames_total": 0,
        "frames_valid": 0,
        "strata_all": {},
        "strata_valid": {},
        "strata_invalid": {},
        "per_seq": {},
    }
    for i in range(len(ds)):
        d = ds[i]
        kp2d = d["kp2d"]
        conf = kp2d[..., 2]
        L = conf.shape[0]
        mask = d.get("mask", torch.ones(L, dtype=torch.bool))
        mask = mask.bool()

        s_all = strata_counts(conf)
        s_val = strata_counts(conf[mask])
        s_inv = strata_counts(conf[~mask]) if (~mask).any() else {"clean": 0, "mild": 0, "heavy": 0}

        out["frames_total"] += L
        out["frames_valid"] += int(mask.sum())
        out["strata_all"] = merge(out["strata_all"], s_all)
        out["strata_valid"] = merge(out["strata_valid"], s_val)
        out["strata_invalid"] = merge(out["strata_invalid"], s_inv)
        vid = d["meta"]["vid"]
        out["per_seq"][vid] = {"L": L, "valid": int(mask.sum()), **{f"{k}_valid": v for k, v in s_val.items()}}

    # ---- console summary ----
    print(f"\n=== {name} ===")
    print(f"  sequences          : {out['n_seqs']}")
    print(f"  frames total/valid : {out['frames_total']} / {out['frames_valid']}")
    if name in OFFICIAL_MINUTES:
        exp = OFFICIAL_MINUTES[name] * 60 * 30
        print(f"  official duration  : {OFFICIAL_MINUTES[name]} min @30fps ≈ {exp:.0f} frames "
              f"(total/expected = {out['frames_total']/exp:.2f})")
    for key in ["strata_all", "strata_valid", "strata_invalid"]:
        s = out[key]
        tot = sum(s.values()) or 1
        print(f"  {key:15s}: clean {s.get('clean',0)} ({s.get('clean',0)/tot:.1%})  "
              f"mild {s.get('mild',0)} ({s.get('mild',0)/tot:.2%})  "
              f"heavy {s.get('heavy',0)} ({s.get('heavy',0)/tot:.2%})")
    if name in APPENDIX_TABLE2:
        ap = APPENDIX_TABLE2[name]
        print(f"  appendix Table 2   : frames {ap['frames']}  "
              f"clean {ap['clean']:.1%}  mild {ap['mild']:.2%}  heavy {ap['heavy']:.2%}"
              f"   -> frames match: {ap['frames'] in (out['frames_total'], out['frames_valid'])}")
    # heavy frames lost to the valid mask (caveat for the paper)
    lost = out["strata_invalid"].get("heavy", 0)
    if lost:
        print(f"  !! {lost} heavy frames are excluded by the valid mask (truncation-by-construction)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="rich,emdb1,emdb2,3dpw")
    args = ap.parse_args()
    wanted = args.datasets.split(",")

    report = {}
    if "rich" in wanted:
        from hmr4d.dataset.rich.rich_motion_test import RichSmplFullSeqDataset
        report["RICH"] = check_dataset("RICH", RichSmplFullSeqDataset())
    if "emdb1" in wanted:
        from hmr4d.dataset.emdb.emdb_motion_test import EmdbSmplFullSeqDataset
        report["EMDB_1"] = check_dataset("EMDB_1", EmdbSmplFullSeqDataset(split=1))
    if "emdb2" in wanted:
        from hmr4d.dataset.emdb.emdb_motion_test import EmdbSmplFullSeqDataset
        report["EMDB_2"] = check_dataset("EMDB_2", EmdbSmplFullSeqDataset(split=2))
    if "3dpw" in wanted:
        from hmr4d.dataset.threedpw.threedpw_motion_test import ThreedpwSmplFullSeqDataset
        report["3DPW"] = check_dataset("3DPW", ThreedpwSmplFullSeqDataset(flip_test=False))

    out_dir = Path("results/e0_integrity")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nFull report -> {out_dir / 'report.json'}")


if __name__ == "__main__":
    main()
