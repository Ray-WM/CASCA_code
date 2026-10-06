"""Per-sequence stress deltas for bootstrap CIs (re-review NM2).

Runs the e8 corruption at p in {0, 0.4} only, for one model, and dumps
PER-SEQUENCE WA-MPJPE so the paired control-vs-CASCA degradation gap can
be bootstrapped locally. Protocol identical to tools/e8_stress.py
(same seeds, bursts, postproc).

  python tools/e8_perseq.py exp=gvhmr/mixed/mixed ckpt_path=<ckpt> +e8_tag=control
Output: results/e8_perseq/<tag>.json  {ds: {vid: {"p0": wa, "p40": wa}}}
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

RATES = [0.0, 0.40]
BURST = 5
OFFSET = 0.3
GSTD = 0.15


def corrupt(obs, rate, gen):
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


@hydra.main(version_base="1.3", config_path="../hmr4d/configs", config_name="train")
def main(cfg: DictConfig) -> None:
    from hmr4d.dataset.emdb.emdb_motion_test import EmdbSmplFullSeqDataset
    from hmr4d.dataset.rich.rich_motion_test import RichSmplFullSeqDataset
    from hmr4d.utils.eval.eval_utils import compute_global_metrics
    from hmr4d.utils.geo.hmr_cam import normalize_kp2d
    from hmr4d.utils.geo_transform import apply_T_on_points
    from hmr4d.utils.net_utils import load_pretrained_model
    from hmr4d.utils.smplx_utils import make_smplx

    tag = cfg.get("e8_tag", "model")
    out_fn = Path("results/e8_perseq") / f"{tag}.json"
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
    res = {ds: {} for ds in datasets}

    with torch.no_grad():
        for ds_name, ds in datasets.items():
            for i in range(len(ds)):
                d = ds[i]
                gender, L = d["gender"], d["length"]
                mask = d["mask"].bool() if "mask" in d else None
                if ds_name == "RICH":
                    gt_p = {k: v.cuda() for k, v in d["gt_smpl_params"].items()}
                    w_v = torch.stack([torch.matmul(s2s, x) for x in smplx_rich[gender](**gt_p).vertices])
                    t_ay_v = apply_T_on_points(w_v, d["T_w2ay"].cuda())
                    pred_body = smplx_rich["neutral"]
                else:
                    gt_p = {k: v.cuda() for k, v in d["smpl_params"].items()}
                    t_ay_v = smpl_g[gender](**gt_p).vertices
                    pred_body = smplx_pred

                batch = {k: d[k][None].cuda().float() for k in ["kp2d", "bbx_xys", "cam_angvel", "f_imgseq"]}
                K = d["K_fullimg"]
                batch["K_fullimg"] = (K if K.dim() == 3 else K[None].repeat(L, 1, 1))[None].cuda().float()
                obs0 = normalize_kp2d(batch["kp2d"], batch["bbx_xys"])
                if mask is not None:
                    obs0[0, ~mask.cuda()] = 0

                entry = {}
                for r in RATES:
                    gen = torch.Generator(device=obs0.device).manual_seed(7000 + i * 100 + int(r * 100))
                    obs = corrupt(obs0, r, gen)
                    out = model.pipeline.forward(
                        dict(length=torch.tensor([L]).cuda(), obs=obs, bbx_xys=batch["bbx_xys"],
                             K_fullimg=batch["K_fullimg"], cam_angvel=batch["cam_angvel"],
                             f_imgseq=batch["f_imgseq"]), train=False, postproc=True)
                    glo = {k: (v[0] if isinstance(v, torch.Tensor) and v.dim() == 3 else v)
                           for k, v in out["pred_smpl_params_global"].items()}
                    p_ay_v = torch.stack([torch.matmul(s2s, x) for x in pred_body(**glo).vertices])
                    g = compute_global_metrics(
                        dict(pred_j3d_glob=jj(p_ay_v), target_j3d_glob=jj(t_ay_v),
                             pred_verts_glob=p_ay_v, target_verts_glob=t_ay_v), mask=mask)
                    entry["p0" if r == 0 else "p40"] = float(np.atleast_1d(g["waa_mpjpe"]).mean())
                res[ds_name][d["meta"]["vid"]] = entry
                Log.info(f"[E8ps] {tag} {ds_name} {i+1}/{len(ds)}")
            with open(out_fn, "w") as f:
                json.dump(res, f)
    Log.info(f"[E8ps] done -> {out_fn}")


if __name__ == "__main__":
    register_store_gvhmr()
    main()
