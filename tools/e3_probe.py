"""E3: controlled noise-injection probe on EMDB-1 (TNP evidence, re-run on
retrained checkpoints; the original remote-only script was lost).

For each EMDB-1 sequence, corrupt a 5-frame window at the sequence midpoint
under one of three perturbations and record per-frame camera-space MPJPE for
the clean and corrupted runs:
  zeroconf : set all keypoint confidences in the window to 0 (gate-visible)
  offset   : add +0.3 (normalized units) to keypoint coords (gate-blind)
  gauss    : add N(0, 0.3) noise to keypoint coords (gate-blind)

Output JSON (compatible with tools/plot_e1.py):
  results/e3_probe/<tag>_{clean,zeroconf,offset,gauss}_emdb1.json
  { vid: {"mpjpe": [...], "valid": [...], "noise_start": s, "noise_end": e} }
plus a spillover summary (mean dMPJPE over offsets +1..+7) per perturbation.

Run on the REMOTE machine (1 GPU):
  python tools/e3_probe.py exp=gvhmr/mixed/mixed ckpt_path=<control.ckpt> +e3_tag=control
  python tools/e3_probe.py exp=gvhmr/mixed/casca ckpt_path=<casca.ckpt> +e3_tag=casca
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

WINDOW = 5
PERTURBS = ["clean", "zeroconf", "offset", "gauss"]
OFFSET_MAG = 0.3
GAUSS_STD = 0.3


def perturb_obs(obs, mode, lo, hi, gen):
    """obs: (1, L, 17, 3) normalized kp2d+conf. Perturb frames [lo, hi)."""
    obs = obs.clone()
    if mode == "zeroconf":
        obs[0, lo:hi, :, 2] = 0.0
    elif mode == "offset":
        obs[0, lo:hi, :, :2] += OFFSET_MAG
    elif mode == "gauss":
        noise = torch.randn(hi - lo, obs.shape[2], 2, generator=gen, device=obs.device)
        obs[0, lo:hi, :, :2] += GAUSS_STD * noise
    return obs


@hydra.main(version_base="1.3", config_path="../hmr4d/configs", config_name="train")
def main(cfg: DictConfig) -> None:
    from hmr4d.dataset.emdb.emdb_motion_test import EmdbSmplFullSeqDataset
    from hmr4d.utils.eval.eval_utils import batch_align_by_pelvis, compute_jpe
    from hmr4d.utils.geo.hmr_cam import normalize_kp2d
    from hmr4d.utils.geo_transform import apply_T_on_points
    from hmr4d.utils.net_utils import load_pretrained_model
    from hmr4d.utils.smplx_utils import make_smplx

    tag = cfg.get("e3_tag", "model")
    out_dir = Path("results/e3_probe")
    out_dir.mkdir(parents=True, exist_ok=True)

    model = hydra.utils.instantiate(cfg.model, _recursive_=False)
    assert cfg.ckpt_path is not None, "pass ckpt_path=..."
    load_pretrained_model(model, cfg.ckpt_path)
    model = model.cuda().eval()

    smplx_pred = make_smplx("supermotion").cuda()
    smpl_gt = {g: make_smplx("smpl", gender=g).cuda() for g in ["male", "female"]}
    J24 = torch.load("hmr4d/utils/body_model/smpl_neutral_J_regressor.pt").cuda()
    smplx2smpl = torch.load("hmr4d/utils/body_model/smplx2smpl_sparse.pt").cuda()

    ds = EmdbSmplFullSeqDataset(split=1)
    results = {m: {} for m in PERTURBS}

    with torch.no_grad():
        for i in range(len(ds)):
            d = ds[i]
            vid, L = d["meta"]["vid"], d["length"]
            mid = L // 2
            lo, hi = mid - WINDOW // 2, mid - WINDOW // 2 + WINDOW

            # GT camera-space joints
            params = {k: v.cuda() for k, v in d["smpl_params"].items()}
            w_verts = smpl_gt[d["gender"]](**params).vertices
            t_j24 = einsum(J24, apply_T_on_points(w_verts, d["T_w2c"].cuda()), "j v, l v i -> l j i")

            batch = {k: d[k][None].cuda().float() for k in ["kp2d", "bbx_xys", "cam_angvel", "f_imgseq"]}
            batch["K_fullimg"] = d["K_fullimg"][None].cuda().float()
            length = torch.tensor([L]).cuda()
            obs0 = normalize_kp2d(batch["kp2d"], batch["bbx_xys"])
            mask = d["mask"].bool().cuda()
            obs0[0, ~mask] = 0
            gen = torch.Generator(device=obs0.device).manual_seed(1000 + i)

            for mode in PERTURBS:
                obs = obs0 if mode == "clean" else perturb_obs(obs0, mode, lo, hi, gen)
                out = model.pipeline.forward(
                    dict(length=length, obs=obs, bbx_xys=batch["bbx_xys"],
                         K_fullimg=batch["K_fullimg"], cam_angvel=batch["cam_angvel"],
                         f_imgseq=batch["f_imgseq"]),
                    train=False, postproc=False,
                )
                pred = {k: v[0] for k, v in out["pred_smpl_params_incam"].items()}
                p_verts = torch.stack([torch.matmul(smplx2smpl, v_) for v_ in smplx_pred(**pred).vertices])
                p_j24 = einsum(J24, p_verts, "j v, l v i -> l j i")
                a, b, _, _ = batch_align_by_pelvis(
                    [p_j24.cpu(), t_j24.cpu(), p_j24.cpu().clone(), t_j24.cpu().clone()], [1, 2]
                )
                mpjpe = (compute_jpe(a, b) * 1000.0).tolist()
                results[mode][vid] = {
                    "mpjpe": mpjpe,
                    "valid": mask.cpu().numpy().astype(int).tolist(),
                    "noise_start": int(lo),
                    "noise_end": int(hi - 1),
                }
            Log.info(f"[E3] {i+1}/{len(ds)} {vid} done (L={L}, window {lo}-{hi-1})")

    # save per-perturbation JSONs
    for mode in PERTURBS:
        fn = out_dir / f"{tag}_{mode}_emdb1.json"
        with open(fn, "w") as f:
            json.dump(results[mode], f)
        Log.info(f"[E3] wrote {fn}")

    # spillover summary: mean dMPJPE over offsets +1..+7 (valid frames only)
    summary = {}
    for mode in PERTURBS[1:]:
        deltas = []
        for vid, noisy in results[mode].items():
            clean = results["clean"][vid]
            c, n = np.array(clean["mpjpe"]), np.array(noisy["mpjpe"])
            v = np.array(noisy["valid"], dtype=bool)
            e = noisy["noise_end"]
            for off in range(1, 8):
                t = e + off
                if t < len(c) and v[t]:
                    deltas.append(n[t] - c[t])
        summary[mode] = float(np.mean(deltas)) if deltas else None
    summary_fn = out_dir / f"{tag}_spillover.json"
    with open(summary_fn, "w") as f:
        json.dump(summary, f, indent=2)
    Log.info(f"[E3] spillover ({tag}): {summary} -> {summary_fn}")


if __name__ == "__main__":
    register_store_gvhmr()
    main()
