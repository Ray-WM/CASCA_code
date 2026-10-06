"""E11: realistic structured-corruption stress test (post-internal-review).

Reviewers' external-validity ask (R2 W3 option c / R1 MW2 b): corruption
modeled on REAL detector failure modes rather than arbitrary noise, while
remaining gate-blind (confidences untouched -> confidently-wrong failures).

Three modes, applied in 5-frame bursts at random positions (same burst/rate
machinery and seed policy as tools/e8_stress.py, hence directly comparable):

  lrswap : whole-body left<->right joint swap (detector flips facing
           direction; classic back-view failure).
  stuck  : all joint coords frozen at the pre-burst frame (detector locks
           onto occluder/background while the person moves).
  hijack : lower-body joints (hips/knees/ankles) coherently displaced by a
           random direction, 0.15-0.30 bbox-unit magnitude (joints snap to
           an occluder or nearby person during partial occlusion).

Rates: p in {0.0, 0.10, 0.40}; one deterministic realization per
(sequence, mode, rate). Full official metric suite (postproc on).

Usage (remote, 1 GPU per model):
  python tools/e11_realistic_stress.py exp=gvhmr/mixed/mixed ckpt_path=<ckpt> +e11_tag=control
Output: results/e11_realistic/<tag>.json  {rate: {mode: {dataset: metrics}}}
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

RATES = [0.0, 0.10, 0.40]
MODES = ["lrswap", "stuck", "hijack"]
BURST = 5
# COCO17 left/right pairs (eyes, ears, shoulders, elbows, wrists, hips, knees, ankles)
LR_L = [1, 3, 5, 7, 9, 11, 13, 15]
LR_R = [2, 4, 6, 8, 10, 12, 14, 16]
LOWER = [11, 12, 13, 14, 15, 16]
HIJACK_MAG = (0.15, 0.30)  # bbox-normalized units


def burst_starts(L, rate, gen, device):
    n_bursts = max(1, int(round(rate * L / BURST)))
    return torch.randint(0, max(1, L - BURST), (n_bursts,), generator=gen, device=device).tolist()


def corrupt(obs, mode, rate, gen):
    """Gate-blind structured corruption: coords only, conf untouched."""
    if rate <= 0:
        return obs
    obs = obs.clone()
    L = obs.shape[1]
    for s in burst_starts(L, rate, gen, obs.device):
        e = min(L, s + BURST)
        if mode == "lrswap":
            left = obs[0, s:e, LR_L, :2].clone()
            obs[0, s:e, LR_L, :2] = obs[0, s:e, LR_R, :2]
            obs[0, s:e, LR_R, :2] = left
        elif mode == "stuck":
            src = max(0, s - 1)
            obs[0, s:e, :, :2] = obs[0, src, :, :2].unsqueeze(0)
        elif mode == "hijack":
            theta = torch.rand(1, generator=gen, device=obs.device) * 2 * np.pi
            mag = HIJACK_MAG[0] + torch.rand(1, generator=gen, device=obs.device) * (
                HIJACK_MAG[1] - HIJACK_MAG[0])
            off = torch.cat([torch.cos(theta), torch.sin(theta)]) * mag
            obs[0, s:e][:, LOWER, :2] += off
        else:
            raise ValueError(mode)
    return obs


@hydra.main(version_base="1.3", config_path="../hmr4d/configs", config_name="train")
def main(cfg: DictConfig) -> None:
    from hmr4d.dataset.emdb.emdb_motion_test import EmdbSmplFullSeqDataset
    from hmr4d.dataset.rich.rich_motion_test import RichSmplFullSeqDataset
    from hmr4d.utils.eval.eval_utils import compute_camcoord_metrics, compute_global_metrics
    from hmr4d.utils.geo.hmr_cam import normalize_kp2d
    from hmr4d.utils.geo_transform import apply_T_on_points
    from hmr4d.utils.net_utils import load_pretrained_model
    from hmr4d.utils.smplx_utils import make_smplx

    tag = cfg.get("e11_tag", "model")
    out_fn = Path("results/e11_realistic") / f"{tag}.json"
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
    METRICS = ["mpjpe", "wa", "w", "jitter", "fs", "rte", "accel"]
    # rate 0.0 is mode-independent: store under mode "clean"
    cells = [("0.00", "clean")] + [(f"{r:.2f}", m) for r in RATES if r > 0 for m in MODES]
    agg = {}
    for rk, mk in cells:
        agg.setdefault(rk, {})[mk] = {ds: {m: [] for m in METRICS} for ds in datasets}

    with torch.no_grad():
        for ds_name, ds in datasets.items():
            for i in range(len(ds)):
                d = ds[i]
                gender, L = d["gender"], d["length"]
                mask = d["mask"].bool() if "mask" in d else None
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

                for rk, mk in cells:
                    r = float(rk)
                    gen = torch.Generator(device=obs0.device).manual_seed(
                        11000 + i * 100 + int(r * 100) + MODES.index(mk) * 7 if mk != "clean" else 11000 + i * 100)
                    obs = obs0 if mk == "clean" else corrupt(obs0, mk, r, gen)
                    out = model.pipeline.forward(
                        dict(length=torch.tensor([L]).cuda(), obs=obs, bbx_xys=batch["bbx_xys"],
                             K_fullimg=batch["K_fullimg"], cam_angvel=batch["cam_angvel"],
                             f_imgseq=batch["f_imgseq"]), train=False, postproc=True)
                    sq = lambda p: {k: (v[0] if isinstance(v, torch.Tensor) and v.dim() == 3 else v)
                                    for k, v in p.items()}
                    inc_t, glo_t = sq(out["pred_smpl_params_incam"]), sq(out["pred_smpl_params_global"])
                    p_c_v = torch.stack([torch.matmul(s2s, x) for x in pred_body(**inc_t).vertices])
                    p_ay_v = torch.stack([torch.matmul(s2s, x) for x in pred_body(**glo_t).vertices])
                    cam = compute_camcoord_metrics(
                        dict(pred_j3d=jj(p_c_v), target_j3d=jj(t_c_v),
                             pred_verts=p_c_v, target_verts=t_c_v), mask=mask)
                    glob = compute_global_metrics(
                        dict(pred_j3d_glob=jj(p_ay_v), target_j3d_glob=jj(t_ay_v),
                             pred_verts_glob=p_ay_v, target_verts_glob=t_ay_v), mask=mask)
                    a = agg[rk][mk][ds_name]
                    a["mpjpe"].append(cam["mpjpe"]); a["accel"].append(np.atleast_1d(cam["accel"]))
                    a["wa"].append(np.atleast_1d(glob["waa_mpjpe"])); a["w"].append(np.atleast_1d(glob["wa2_mpjpe"]))
                    a["jitter"].append(np.atleast_1d(glob["jitter"])); a["fs"].append(np.atleast_1d(glob["fs"]))
                    a["rte"].append(np.atleast_1d(glob["rte"]))
                Log.info(f"[E11] {tag} {ds_name} {i+1}/{len(ds)}")
            res = {rk: {mk: {ds: {m: (float(np.concatenate(v).mean()) if v else None)
                                  for m, v in agg[rk][mk][ds].items()} for ds in agg[rk][mk]}
                        for mk in agg[rk]} for rk in agg}
            with open(out_fn, "w") as f:
                json.dump(res, f, indent=2)
    Log.info(f"[E11] done -> {out_fn}")


if __name__ == "__main__":
    register_store_gvhmr()
    main()
