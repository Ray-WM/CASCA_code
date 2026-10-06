"""E1: post-hoc smoothing control, clean re-designed protocol (2026-08).

Stage 1 (infer): run the control model over RICH + EMDB-2 with the standard
test-time post-processing, cache predicted SMPL params (incam + global) and
per-frame low-conf counts per sequence to results/e1_cache/.
Stage 2 (sweep): for each filter setting, smooth the cached parameter
sequences (transl linearly; rotations via 6D -> filter -> re-orthonormalize),
recompute the FULL metric suite through the same eval path as the main
tables, and dump the whole grid to results/e1_smoothing/sweep.json.

Filters and grids:
  savgol : window in {5, 9, 15, 21, 31}, polyorder 2 (window > polyorder)
  oneeuro: (min_cutoff, beta) in {0.1,0.3,1.0} x {0.01,0.1,0.3}

Usage (remote, 1 GPU):
  python tools/e1_smoothing.py exp=gvhmr/mixed/mixed ckpt_path=<control.ckpt>
  # stage 2 reuses the cache automatically on re-runs
"""

import itertools
import json
from pathlib import Path

import hydra
import numpy as np
import torch
from einops import einsum
from omegaconf import DictConfig

from hmr4d.configs import register_store_gvhmr
from hmr4d.utils.pylogger import Log

CACHE = Path("results/e1_cache")
OUT = Path("results/e1_smoothing")
SETTINGS = (
    [("none", {})]
    + [("savgol", {"window": w, "poly": 2}) for w in [5, 9, 15, 21, 31]]
    + [("oneeuro", {"mc": mc, "beta": b}) for mc, b in itertools.product([0.1, 0.3, 1.0], [0.01, 0.1, 0.3])]
)


# ---------------- filters ----------------
def savgol_np(x, window, poly):
    from scipy.signal import savgol_filter

    if len(x) <= window:
        return x
    return savgol_filter(x, window, poly, axis=0)


def oneeuro_np(x, mc, beta, freq=30.0, dcutoff=1.0):
    def alpha(cutoff):
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau * freq)

    y = np.empty_like(x)
    y[0] = x[0]
    dx_prev = np.zeros_like(x[0])
    for t in range(1, len(x)):
        dx = (x[t] - y[t - 1]) * freq
        dx_hat = alpha(dcutoff) * dx + (1 - alpha(dcutoff)) * dx_prev
        cutoff = mc + beta * np.abs(dx_hat)
        a = 1.0 / (1.0 + (freq / (2 * np.pi * cutoff)))
        y[t] = a * x[t] + (1 - a) * y[t - 1]
        dx_prev = dx_hat
    return y


def smooth_params(params, mode, cfg):
    """params: dict of np arrays (L, D). Rotations filtered in 6D."""
    from pytorch3d.transforms import (axis_angle_to_matrix, matrix_to_axis_angle,
                                      matrix_to_rotation_6d, rotation_6d_to_matrix)

    if mode == "none":
        return params
    f = (lambda x: savgol_np(x, cfg["window"], cfg["poly"])) if mode == "savgol" else (
        lambda x: oneeuro_np(x, cfg["mc"], cfg["beta"]))

    out = {}
    for k, v in params.items():
        if k == "betas":
            out[k] = v
        elif k == "transl":
            out[k] = f(v.copy())
        else:  # global_orient (L,3) or body_pose (L,63): axis-angle joints
            aa = torch.from_numpy(v).float().reshape(len(v), -1, 3)
            r6 = matrix_to_rotation_6d(axis_angle_to_matrix(aa)).numpy()  # (L,J,6)
            r6 = f(r6.reshape(len(v), -1)).reshape(r6.shape)
            m = rotation_6d_to_matrix(torch.from_numpy(r6).float())
            out[k] = matrix_to_axis_angle(m).reshape(len(v), -1).numpy()
    return out


# ---------------- main ----------------
@hydra.main(version_base="1.3", config_path="../hmr4d/configs", config_name="train")
def main(cfg: DictConfig) -> None:
    from hmr4d.dataset.emdb.emdb_motion_test import EmdbSmplFullSeqDataset
    from hmr4d.dataset.rich.rich_motion_test import RichSmplFullSeqDataset
    from hmr4d.utils.eval.eval_utils import (batch_align_by_pelvis, compute_camcoord_metrics,
                                             compute_global_metrics, compute_jpe)
    from hmr4d.utils.geo.hmr_cam import normalize_kp2d
    from hmr4d.utils.geo_transform import apply_T_on_points
    from hmr4d.utils.net_utils import load_pretrained_model
    from hmr4d.utils.smplx_utils import make_smplx

    global CACHE
    tag = cfg.get("e1_tag", "")
    sweep_name = f"sweep_{tag}.json" if tag else "sweep.json"
    if tag:
        CACHE = Path(f"results/e1_cache_{tag}")
    CACHE.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)

    smplx_rich = {g: make_smplx("rich-smplx", gender=g).cuda() for g in ["male", "female", "neutral"]}
    smpl_g = {g: make_smplx("smpl", gender=g).cuda() for g in ["male", "female"]}
    smplx_pred = make_smplx("supermotion").cuda()
    J24 = torch.load("hmr4d/utils/body_model/smpl_neutral_J_regressor.pt").cuda()
    s2s = torch.load("hmr4d/utils/body_model/smplx2smpl_sparse.pt").cuda()

    datasets = {"RICH": RichSmplFullSeqDataset(), "EMDB_2": EmdbSmplFullSeqDataset(split=2)}

    # ---------- Stage 1: inference cache ----------
    need = [ (ds, i) for ds, d in datasets.items() for i in range(len(d))
             if not (CACHE / ds / f"{i:04d}.npz").exists() ]
    if need:
        model = hydra.utils.instantiate(cfg.model, _recursive_=False)
        assert cfg.ckpt_path is not None
        load_pretrained_model(model, cfg.ckpt_path)
        model = model.cuda().eval()
        with torch.no_grad():
            for ds_name, i in need:
                d = datasets[ds_name][i]
                (CACHE / ds_name).mkdir(parents=True, exist_ok=True)
                batch = {k: d[k][None].cuda().float() for k in ["kp2d", "bbx_xys", "cam_angvel", "f_imgseq"]}
                K = d["K_fullimg"]
                batch["K_fullimg"] = (K if K.dim() == 3 else K[None].repeat(d["length"], 1, 1))[None].cuda().float()
                obs = normalize_kp2d(batch["kp2d"], batch["bbx_xys"])
                if "mask" in d:
                    obs[0, ~d["mask"].bool().cuda()] = 0
                out = model.pipeline.forward(
                    dict(length=torch.tensor([d["length"]]).cuda(), obs=obs,
                         bbx_xys=batch["bbx_xys"], K_fullimg=batch["K_fullimg"],
                         cam_angvel=batch["cam_angvel"], f_imgseq=batch["f_imgseq"]),
                    train=False, postproc=True)  # static_cam stays False: matches the main-table eval protocol
                save = {"vid": d["meta"]["vid"],
                        "nlow": (d["kp2d"][..., 2] < 0.5).sum(-1).numpy()}
                for pk, pref in [("pred_smpl_params_incam", "c_"), ("pred_smpl_params_global", "g_")]:
                    for k, v in out[pk].items():
                        save[pref + k] = v[0].cpu().numpy()
                if "mask" in d:
                    save["valid"] = d["mask"].numpy()
                np.savez_compressed(CACHE / ds_name / f"{i:04d}.npz", **save)
                Log.info(f"[E1 cache] {ds_name} {i+1}/{len(datasets[ds_name])}")

    # ---------- Stage 2: sweep ----------
    results = {}
    for mode, fcfg in SETTINGS:
        key = mode + ("" if not fcfg else "_" + "_".join(f"{k}{v}" for k, v in fcfg.items()))
        agg = {ds: {m: [] for m in ["mpjpe", "mpjpe_mild", "wa", "w", "jitter", "fs", "accel", "rte"]}
               for ds in datasets}
        for ds_name, ds in datasets.items():
            for i in range(len(ds)):
                z = np.load(CACHE / ds_name / f"{i:04d}.npz", allow_pickle=True)
                d = ds[i]
                gender = d["gender"]
                mask = torch.from_numpy(z["valid"]).bool() if "valid" in z else None

                inc = smooth_params({k[2:]: z[k] for k in z if k.startswith("c_")}, mode, fcfg)
                glo = smooth_params({k[2:]: z[k] for k in z if k.startswith("g_")}, mode, fcfg)
                inc_t = {k: torch.from_numpy(v).float().cuda() for k, v in inc.items()}
                glo_t = {k: torch.from_numpy(v).float().cuda() for k, v in glo.items()}

                with torch.no_grad():
                    if ds_name == "RICH":
                        gt_p = {k: v.cuda() for k, v in d["gt_smpl_params"].items()}
                        w_v = torch.stack([torch.matmul(s2s, x) for x in smplx_rich[gender](**gt_p).vertices])
                        t_c_v = apply_T_on_points(w_v, d["T_w2c"].cuda())
                        t_ay_v = apply_T_on_points(w_v, d["T_w2ay"].cuda())
                        p_c_v = torch.stack([torch.matmul(s2s, x) for x in smplx_rich["neutral"](**inc_t).vertices])
                        p_ay_v = torch.stack([torch.matmul(s2s, x) for x in smplx_rich["neutral"](**glo_t).vertices])
                    else:
                        gt_p = {k: v.cuda() for k, v in d["smpl_params"].items()}
                        w_v = smpl_g[gender](**gt_p).vertices
                        t_c_v = apply_T_on_points(w_v, d["T_w2c"].cuda())
                        t_ay_v = w_v
                        p_c_v = torch.stack([torch.matmul(s2s, x) for x in smplx_pred(**inc_t).vertices])
                        p_ay_v = torch.stack([torch.matmul(s2s, x) for x in smplx_pred(**glo_t).vertices])
                    jj = lambda vv: einsum(J24, vv, "j v, l v i -> l j i")
                    cam = compute_camcoord_metrics(
                        dict(pred_j3d=jj(p_c_v), target_j3d=jj(t_c_v),
                             pred_verts=p_c_v, target_verts=t_c_v), mask=mask)
                    glob = compute_global_metrics(
                        dict(pred_j3d_glob=jj(p_ay_v), target_j3d_glob=jj(t_ay_v),
                             pred_verts_glob=p_ay_v, target_verts_glob=t_ay_v), mask=mask)
                nlow = z["nlow"] if mask is None else z["nlow"][mask.numpy()]
                agg[ds_name]["mpjpe"].append(cam["mpjpe"])
                mild = (nlow >= 3) & (nlow < 6)
                if mild.any():
                    agg[ds_name]["mpjpe_mild"].append(cam["mpjpe"][mild])
                agg[ds_name]["accel"].append(np.atleast_1d(cam["accel"]))
                agg[ds_name]["wa"].append(np.atleast_1d(glob["waa_mpjpe"]))
                agg[ds_name]["w"].append(np.atleast_1d(glob["wa2_mpjpe"]))
                agg[ds_name]["jitter"].append(np.atleast_1d(glob["jitter"]))
                agg[ds_name]["fs"].append(np.atleast_1d(glob["fs"]))
                agg[ds_name]["rte"].append(np.atleast_1d(glob["rte"]))
        results[key] = {ds: {m: float(np.concatenate(v).mean()) if v else None
                             for m, v in agg[ds].items()} for ds in datasets}
        r = results[key]["RICH"]
        Log.info(f"[E1] {key:24s} RICH mpjpe={r['mpjpe']:.1f} mild={r['mpjpe_mild']:.1f} "
                 f"wa={r['wa']:.1f} jitter={r['jitter']:.2f} fs={r['fs']:.2f}")
        with open(OUT / sweep_name, "w") as f:
            json.dump(results, f, indent=2)
    Log.info(f"[E1] sweep done -> {OUT/sweep_name}")


if __name__ == "__main__":
    register_store_gvhmr()
    main()
