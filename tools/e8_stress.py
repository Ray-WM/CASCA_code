"""E8-stress: system-level corruption stress test (the decisive experiment).

Corrupt a fraction p of frames across each RICH / EMDB-2 sequence with
gate-blind coordinate noise (5-frame bursts of constant offset + Gaussian
jitter at random locations), then compute the FULL metric suite (same eval
path as the main tables, postproc included) as a function of p, for the
control model and CASCA.  A smoothing arm (best Savitzky-Golay from E1)
is applied to the corrupted control outputs to test whether post-hoc
filtering rescues degraded inputs.

Rates: p in {0, 0.05, 0.10, 0.20, 0.40} (fraction of frames corrupted).
Burst corruption: coords += 0.3 offset + N(0, 0.15); conf/angvel untouched.

Usage (remote, 1 GPU per model):
  python tools/e8_stress.py exp=gvhmr/mixed/mixed ckpt_path=<control.ckpt> +e8_tag=control
  python tools/e8_stress.py exp=gvhmr/mixed/casca ckpt_path=<casca.ckpt> +e8_tag=casca
Output: results/e8_stress/<tag>.json  (metrics per dataset per rate,
        with and without SG-w9 output smoothing)
"""

import json
from pathlib import Path

import hydra
import numpy as np
import torch
from einops import einsum
from omegaconf import DictConfig

from hmr4d.configs import register_store_gvhmr
from hmr4d.utils.pylogger import Log

RATES = [0.0, 0.05, 0.10, 0.20, 0.40]
BURST = 5
OFFSET = 0.3
GSTD = 0.15
SG_WINDOW, SG_POLY = 9, 2


def corrupt(obs, rate, gen):
    """Corrupt ~rate of frames in 5-frame bursts (gate-blind: coords only)."""
    if rate <= 0:
        return obs
    obs = obs.clone()
    L = obs.shape[1]
    n_bursts = max(1, int(round(rate * L / BURST)))
    starts = torch.randint(0, max(1, L - BURST), (n_bursts,), generator=gen, device=obs.device)
    for s in starts.tolist():
        e = min(L, s + BURST)
        obs[0, s:e, :, :2] += OFFSET
        noise = torch.randn(e - s, obs.shape[2], 2, generator=gen, device=obs.device)
        obs[0, s:e, :, :2] += GSTD * noise
    return obs


def sg_params(params, window=SG_WINDOW, poly=SG_POLY):
    from pytorch3d.transforms import (axis_angle_to_matrix, matrix_to_axis_angle,
                                      matrix_to_rotation_6d, rotation_6d_to_matrix)
    from scipy.signal import savgol_filter

    out = {}
    for k, v in params.items():
        v = v.cpu().numpy()
        if k == "betas":
            out[k] = v
        elif k == "transl":
            out[k] = savgol_filter(v, window, poly, axis=0) if len(v) > window else v
        else:
            aa = torch.from_numpy(v).float().reshape(len(v), -1, 3)
            r6 = matrix_to_rotation_6d(axis_angle_to_matrix(aa)).numpy().reshape(len(v), -1)
            if len(v) > window:
                r6 = savgol_filter(r6, window, poly, axis=0)
            m = rotation_6d_to_matrix(torch.from_numpy(r6.reshape(len(v), -1, 6)).float())
            out[k] = matrix_to_axis_angle(m).reshape(len(v), -1).numpy()
    return {k: torch.from_numpy(np.ascontiguousarray(x)).float().cuda() for k, x in out.items()}


@hydra.main(version_base="1.3", config_path="../hmr4d/configs", config_name="train")
def main(cfg: DictConfig) -> None:
    from hmr4d.dataset.emdb.emdb_motion_test import EmdbSmplFullSeqDataset
    from hmr4d.dataset.rich.rich_motion_test import RichSmplFullSeqDataset
    from hmr4d.utils.eval.eval_utils import compute_camcoord_metrics, compute_global_metrics
    from hmr4d.utils.geo.hmr_cam import normalize_kp2d
    from hmr4d.utils.geo_transform import apply_T_on_points
    from hmr4d.utils.net_utils import load_pretrained_model
    from hmr4d.utils.smplx_utils import make_smplx

    tag = cfg.get("e8_tag", "model")
    do_sg = tag == "control"  # smoothing arm only meaningful for the baseline
    out_fn = Path("results/e8_stress") / f"{tag}.json"
    out_fn.parent.mkdir(parents=True, exist_ok=True)

    model = hydra.utils.instantiate(cfg.model, _recursive_=False)
    assert cfg.ckpt_path is not None
    load_pretrained_model(model, cfg.ckpt_path)
    model = model.cuda().eval()

    smplx_rich = {g: make_smplx("rich-smplx", gender=g).cuda() for g in ["male", "female", "neutral"]}
    smpl_g = {g: make_smplx("smpl", gender=g).cuda() for g in ["male", "female"]}
    smplx_pred = make_smplx("supermotion").cuda()
    J24 = torch.load("hmr4d/utils/body_model/smpl_neutral_J_regressor.pt").cuda()
    s2s = torch.load("hmr4d/utils/body_model/smplx2smpl_sparse.pt").cuda()
    jj = lambda vv: einsum(J24, vv, "j v, l v i -> l j i")

    datasets = {"RICH": RichSmplFullSeqDataset(), "EMDB_2": EmdbSmplFullSeqDataset(split=2)}
    res = {f"{r:.2f}": {a: {ds: {} for ds in datasets} for a in (["raw", "sg"] if do_sg else ["raw"])}
           for r in RATES}
    agg = {k: {a: {ds: {m: [] for m in ["mpjpe", "wa", "w", "jitter", "fs", "rte", "accel"]}
                   for ds in datasets} for a in res[k]} for k in res}

    with torch.no_grad():
        for ds_name, ds in datasets.items():
            for i in range(len(ds)):
                d = ds[i]
                gender, L = d["gender"], d["length"]
                mask = d["mask"].bool() if "mask" in d else None
                # GT (once per sequence)
                if ds_name == "RICH":
                    gt_p = {k: v.cuda() for k, v in d["gt_smpl_params"].items()}
                    w_v = torch.stack([torch.matmul(s2s, x) for x in smplx_rich[gender](**gt_p).vertices])
                    t_c_v = apply_T_on_points(w_v, d["T_w2c"].cuda())
                    t_ay_v = apply_T_on_points(w_v, d["T_w2ay"].cuda())
                    pred_body = smplx_rich["neutral"]
                else:
                    gt_p = {k: v.cuda() for k, v in d["smpl_params"].items()}
                    w_v = smpl_g[gender](**gt_p).vertices
                    t_c_v = apply_T_on_points(w_v, d["T_w2c"].cuda())
                    t_ay_v = w_v
                    pred_body = smplx_pred

                batch = {k: d[k][None].cuda().float() for k in ["kp2d", "bbx_xys", "cam_angvel", "f_imgseq"]}
                K = d["K_fullimg"]
                batch["K_fullimg"] = (K if K.dim() == 3 else K[None].repeat(L, 1, 1))[None].cuda().float()
                obs0 = normalize_kp2d(batch["kp2d"], batch["bbx_xys"])
                if mask is not None:
                    obs0[0, ~mask.cuda()] = 0

                for r in RATES:
                    gen = torch.Generator(device=obs0.device).manual_seed(7000 + i * 100 + int(r * 100))
                    obs = corrupt(obs0, r, gen)
                    out = model.pipeline.forward(
                        dict(length=torch.tensor([L]).cuda(), obs=obs, bbx_xys=batch["bbx_xys"],
                             K_fullimg=batch["K_fullimg"], cam_angvel=batch["cam_angvel"],
                             f_imgseq=batch["f_imgseq"]), train=False, postproc=True)
                    sq = lambda p: {k: (v[0] if isinstance(v, torch.Tensor) and v.dim() == 3 else v)
                                    for k, v in p.items()}
                    raw_inc, raw_glo = sq(out["pred_smpl_params_incam"]), sq(out["pred_smpl_params_global"])
                    arms = {"raw": (raw_inc, raw_glo)}
                    if do_sg:
                        arms["sg"] = (sg_params(raw_inc), sg_params(raw_glo))
                    for arm, (inc_t, glo_t) in arms.items():
                        p_c_v = torch.stack([torch.matmul(s2s, x) for x in pred_body(**inc_t).vertices])
                        p_ay_v = torch.stack([torch.matmul(s2s, x) for x in pred_body(**glo_t).vertices])
                        cam = compute_camcoord_metrics(
                            dict(pred_j3d=jj(p_c_v), target_j3d=jj(t_c_v),
                                 pred_verts=p_c_v, target_verts=t_c_v), mask=mask)
                        glob = compute_global_metrics(
                            dict(pred_j3d_glob=jj(p_ay_v), target_j3d_glob=jj(t_ay_v),
                                 pred_verts_glob=p_ay_v, target_verts_glob=t_ay_v), mask=mask)
                        a = agg[f"{r:.2f}"][arm][ds_name]
                        a["mpjpe"].append(cam["mpjpe"]); a["accel"].append(np.atleast_1d(cam["accel"]))
                        a["wa"].append(np.atleast_1d(glob["waa_mpjpe"])); a["w"].append(np.atleast_1d(glob["wa2_mpjpe"]))
                        a["jitter"].append(np.atleast_1d(glob["jitter"])); a["fs"].append(np.atleast_1d(glob["fs"]))
                        a["rte"].append(np.atleast_1d(glob["rte"]))
                Log.info(f"[E8] {tag} {ds_name} {i+1}/{len(ds)}")
            # persist incrementally
            for rk in agg:
                for arm in agg[rk]:
                    for dsn in agg[rk][arm]:
                        res[rk][arm][dsn] = {m: (float(np.concatenate(v).mean()) if v else None)
                                             for m, v in agg[rk][arm][dsn].items()}
            with open(out_fn, "w") as f:
                json.dump(res, f, indent=2)
    Log.info(f"[E8] done -> {out_fn}")


if __name__ == "__main__":
    register_store_gvhmr()
    main()
