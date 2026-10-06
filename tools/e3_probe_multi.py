"""E3-M: multi-window noise-injection probe (statistical power upgrade).

Extends tools/e3_probe.py after the internal review: the single-window,
17-sequence design gives sequence-level CIs that cross zero. This version
probes K windows per sequence on EMDB-1 (16 usable seqs) and 3DPW test
(37 person-tracks), gate-blind perturbations only (offset, gauss).

Per sequence: 1 clean forward + K windows x 2 modes corrupted forwards.
Windows are evenly spaced in [0.1L, 0.85L], each 5 frames; a window is kept
only if its post-window +1..+7 range fits inside the sequence.

Output: results/e3_probe_multi/<tag>_<dataset>.json
  { vid: {"windows": [ {"start": s, "end_incl": e, "mode": m,
                        "deltas": [<=7 floats]}, ... ] } }
(deltas = corrupted - clean MPJPE (mm) at offsets +1..+7, valid frames only)

Run on the REMOTE machine (1 GPU):
  python tools/e3_probe_multi.py exp=gvhmr/mixed/mixed ckpt_path=<control.ckpt> \
      +e3_tag=control +e3_dataset=emdb1
  python tools/e3_probe_multi.py exp=gvhmr/mixed/casca ckpt_path=<casca.ckpt> \
      +e3_tag=casca +e3_dataset=3dpw
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
TAIL = 7
K_WINDOWS = 8
MODES = ["offset", "gauss"]
OFFSET_MAG = 0.3
GAUSS_STD = 0.3


def perturb_obs(obs, mode, lo, hi, gen):
    obs = obs.clone()
    if mode == "offset":
        obs[0, lo:hi, :, :2] += OFFSET_MAG
    elif mode == "gauss":
        noise = torch.randn(hi - lo, obs.shape[2], 2, generator=gen, device=obs.device)
        obs[0, lo:hi, :, :2] += GAUSS_STD * noise
    return obs


def pick_windows(L):
    starts = np.linspace(int(0.10 * L), int(0.85 * L), K_WINDOWS).astype(int)
    out, last = [], -10**9
    for s in starts:
        if s + WINDOW + TAIL >= L:
            continue
        if s < last + WINDOW + TAIL:  # keep post-windows disjoint from next window
            continue
        out.append(int(s))
        last = s
    return out


@hydra.main(version_base="1.3", config_path="../hmr4d/configs", config_name="train")
def main(cfg: DictConfig) -> None:
    from hmr4d.utils.eval.eval_utils import batch_align_by_pelvis, compute_jpe
    from hmr4d.utils.geo.hmr_cam import normalize_kp2d
    from hmr4d.utils.geo_transform import apply_T_on_points
    from hmr4d.utils.net_utils import load_pretrained_model
    from hmr4d.utils.smplx_utils import make_smplx

    tag = cfg.get("e3_tag", "model")
    ds_name = cfg.get("e3_dataset", "emdb1")
    out_dir = Path("results/e3_probe_multi")
    out_dir.mkdir(parents=True, exist_ok=True)

    model = hydra.utils.instantiate(cfg.model, _recursive_=False)
    assert cfg.ckpt_path is not None, "pass ckpt_path=..."
    load_pretrained_model(model, cfg.ckpt_path)
    model = model.cuda().eval()

    smplx_pred = make_smplx("supermotion").cuda()
    smpl_gt = {g: make_smplx("smpl", gender=g).cuda() for g in ["male", "female"]}
    J24 = torch.load("hmr4d/utils/body_model/smpl_neutral_J_regressor.pt").cuda()
    smplx2smpl = torch.load("hmr4d/utils/body_model/smplx2smpl_sparse.pt").cuda()

    if ds_name == "emdb1":
        from hmr4d.dataset.emdb.emdb_motion_test import EmdbSmplFullSeqDataset
        ds = EmdbSmplFullSeqDataset(split=1)
    elif ds_name == "3dpw":
        from hmr4d.dataset.threedpw.threedpw_motion_test import ThreedpwSmplFullSeqDataset
        ds = ThreedpwSmplFullSeqDataset()
    else:
        raise ValueError(ds_name)

    def run_mpjpe(batch, obs, length, t_j24):
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
        return np.array(compute_jpe(a, b)) * 1000.0

    results = {}
    with torch.no_grad():
        for i in range(len(ds)):
            d = ds[i]
            vid, L = d["meta"]["vid"], d["length"]

            params = {k: v.cuda() for k, v in d["smpl_params"].items()}
            w_verts = smpl_gt[d["gender"]](**params).vertices
            t_j24 = einsum(J24, apply_T_on_points(w_verts, d["T_w2c"].cuda()),
                           "j v, l v i -> l j i")

            batch = {k: d[k][None].cuda().float()
                     for k in ["kp2d", "bbx_xys", "cam_angvel", "f_imgseq"]}
            batch["K_fullimg"] = d["K_fullimg"][None].cuda().float()
            length = torch.tensor([L]).cuda()
            obs0 = normalize_kp2d(batch["kp2d"], batch["bbx_xys"])
            mask = d["mask"].bool().cuda()
            obs0[0, ~mask] = 0
            valid = mask.cpu().numpy().astype(bool)

            mp_clean = run_mpjpe(batch, obs0, length, t_j24)
            windows = pick_windows(L)
            entry = []
            for wi, s in enumerate(windows):
                lo, hi = s, s + WINDOW
                for mode in MODES:
                    gen = torch.Generator(device=obs0.device).manual_seed(
                        5000 + i * 1000 + wi * 10 + MODES.index(mode))
                    obs = perturb_obs(obs0, mode, lo, hi, gen)
                    mp_n = run_mpjpe(batch, obs, length, t_j24)
                    e = hi - 1  # last corrupted frame (inclusive)
                    idx = np.arange(e + 1, min(e + 1 + TAIL, L))
                    idx = idx[valid[idx]]
                    entry.append({
                        "start": int(lo), "end_incl": int(e), "mode": mode,
                        "deltas": (mp_n[idx] - mp_clean[idx]).tolist(),
                    })
            results[vid] = {"windows": entry, "L": int(L), "n_windows": len(windows)}
            Log.info(f"[E3-M] {i+1}/{len(ds)} {vid}: {len(windows)} windows done")

    fn = out_dir / f"{tag}_{ds_name}.json"
    with open(fn, "w") as f:
        json.dump(results, f)
    Log.info(f"[E3-M] wrote {fn}")


if __name__ == "__main__":
    register_store_gvhmr()
    main()
