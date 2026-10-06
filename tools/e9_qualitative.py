"""E9-qual: render qualitative comparison assets on the remote GPU.

For each showcase sequence, run control + CASCA (official postproc) and save:
  - overlay_<model>_f<idx>.png : SMPL mesh over the video frame (subsampled)
  - traj.npz  : top-down root trajectories (pred both models + GT, aligned)
  - curves.npz: per-frame MPJPE (both models) + CASCA d_t
Sequences: RICH Gym_010_cooking1/cam_05 (occlusion), EMDB-2
P4_37_outdoor_run_circle (fast camera), P8_69_outdoor_cartwheel (whole-body).

Usage (remote, 1 GPU):
  python tools/e9_qualitative.py exp=gvhmr/mixed/mixed \
      ckpt_path=<control.ckpt> +ckpt_casca=<casca.ckpt>
Output: results/e9_qualitative/<seq>/...
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

SHOWCASES = [
    ("RICH", "test/Gym_010_dips2/cam_05"),
    ("EMDB_2", "P4_37_outdoor_run_circle"),
    ("EMDB_2", "P8_69_outdoor_cartwheel"),
]
N_OVERLAY = 4  # frames per sequence per model


@hydra.main(version_base="1.3", config_path="../hmr4d/configs", config_name="train")
def main(cfg: DictConfig) -> None:
    from hmr4d.dataset.emdb.emdb_motion_test import EmdbSmplFullSeqDataset
    from hmr4d.dataset.rich.rich_motion_test import RichSmplFullSeqDataset
    from hmr4d.utils.eval.eval_utils import batch_align_by_pelvis, compute_jpe
    from hmr4d.utils.geo.hmr_cam import normalize_kp2d
    from hmr4d.utils.geo_transform import apply_T_on_points
    from hmr4d.utils.net_utils import load_pretrained_model
    from hmr4d.utils.smplx_utils import make_smplx
    from hmr4d.utils.video_io_utils import read_video_np
    from hmr4d.utils.vis.renderer import Renderer
    import imageio

    out_root = Path("results/e9_qualitative")
    ck_ctrl, ck_casca = cfg.ckpt_path, cfg.ckpt_casca

    # two models: control uses current exp cfg (mixed); casca via its exp name
    def build(exp_is_casca, ckpt):
        import copy
        from hydra import compose
        overrides = [f"exp=gvhmr/mixed/{'casca' if exp_is_casca else 'mixed'}"]
        c = compose(config_name="train", overrides=overrides)
        m = hydra.utils.instantiate(c.model, _recursive_=False)
        load_pretrained_model(m, ckpt)
        return m.cuda().eval()

    models = {"control": build(False, ck_ctrl), "casca": build(True, ck_casca)}

    smplx_rich = {g: make_smplx("rich-smplx", gender=g).cuda() for g in ["male", "female", "neutral"]}
    smpl_g = {g: make_smplx("smpl", gender=g).cuda() for g in ["male", "female"]}
    smplx_pred = make_smplx("supermotion").cuda()
    J24 = torch.load("hmr4d/utils/body_model/smpl_neutral_J_regressor.pt").cuda()
    s2s = torch.load("hmr4d/utils/body_model/smplx2smpl_sparse.pt").cuda()
    faces = make_smplx("smpl").faces

    ds_rich = RichSmplFullSeqDataset()
    ds_emdb = EmdbSmplFullSeqDataset(split=2)
    ds_emdb1 = EmdbSmplFullSeqDataset(split=1)

    def find(ds, vid):
        for i in range(len(ds)):
            if vid in ds.idx2meta[i][0]:
                return ds[i]
        raise KeyError(vid)

    with torch.no_grad():
        for ds_name, vid in SHOWCASES:
            seq_dir = out_root / vid.replace("/", "_")
            if (seq_dir / "traj.npz").exists():
                Log.info(f"[E9q] {vid}: already done, skip"); continue
            if ds_name == "RICH":
                d = find(ds_rich, vid)
            else:
                try: d = find(ds_emdb, vid)
                except KeyError: d = find(ds_emdb1, vid)
            seq_dir.mkdir(parents=True, exist_ok=True)
            L = d["length"]
            gender = d["gender"]
            mask = d["mask"].bool() if "mask" in d else torch.ones(L, dtype=torch.bool)

            # GT (camera + world)
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
            gt_traj = einsum(J24, t_ay_v, "j v, l v i -> l j i")[:, 0, :].cpu().numpy()
            gt_verts_c = t_c_v
            gt_pelvis = einsum(J24, t_c_v, "j v, l v i -> l j i")[:, [1, 2], :].mean(1)

            batch = {k: d[k][None].cuda().float() for k in ["kp2d", "bbx_xys", "cam_angvel", "f_imgseq"]}
            K = d["K_fullimg"]
            batch["K_fullimg"] = (K if K.dim() == 3 else K[None].repeat(L, 1, 1))[None].cuda().float()
            obs = normalize_kp2d(batch["kp2d"], batch["bbx_xys"])
            obs[0, ~mask.cuda()] = 0

            res = {}
            for name, model in models.items():
                out = model.pipeline.forward(
                    dict(length=torch.tensor([L]).cuda(), obs=obs, bbx_xys=batch["bbx_xys"],
                         K_fullimg=batch["K_fullimg"], cam_angvel=batch["cam_angvel"],
                         f_imgseq=batch["f_imgseq"]), train=False, postproc=True)
                inc = {k: v[0] for k, v in out["pred_smpl_params_incam"].items()}
                glo = {k: v[0] for k, v in out["pred_smpl_params_global"].items()}
                p_c_v = torch.stack([torch.matmul(s2s, x) for x in pred_body(**inc).vertices])
                p_ay_v = torch.stack([torch.matmul(s2s, x) for x in pred_body(**glo).vertices])
                pj, tj = einsum(J24, p_c_v, "j v, l v i -> l j i"), einsum(J24, t_c_v, "j v, l v i -> l j i")
                a, b, _, _ = batch_align_by_pelvis([pj.cpu(), tj.cpu(), pj.cpu().clone(), tj.cpu().clone()], [1, 2])
                res[name] = dict(
                    mpjpe=(compute_jpe(a, b) * 1000.0),
                    pelvis=pj[:, [1, 2], :].mean(1),
                    traj=einsum(J24, p_ay_v, "j v, l v i -> l j i")[:, 0, :].cpu().numpy(),
                    verts=p_c_v,
                    d_t=(out["model_output"].get("difficulty", torch.full((1, L, 1), np.nan))[0, :, 0].cpu().numpy()),
                )

            # save curves + trajectories
            np.savez_compressed(seq_dir / "curves.npz",
                                mpjpe_control=res["control"]["mpjpe"], mpjpe_casca=res["casca"]["mpjpe"],
                                d_t=res["casca"]["d_t"], valid=mask.numpy())
            np.savez_compressed(seq_dir / "traj.npz",
                                gt=gt_traj, control=res["control"]["traj"], casca=res["casca"]["traj"])

            # overlay frames at most-discriminative valid timestamps
            gap = res["control"]["mpjpe"] - res["casca"]["mpjpe"]
            gap[~mask.numpy()[: len(gap)]] = -1e9
            order = np.argsort(gap)[::-1]
            picks = []
            for t in order:
                if all(abs(int(t) - p) >= 90 for p in picks):
                    picks.append(int(t))
                if len(picks) >= N_OVERLAY:
                    break
            picks = np.array(picks)
            try:
                mr = d["meta_render"]
                try:
                    images = read_video_np(mr["video_path"], scale=mr.get("resize_factor", 0.25))
                except Exception:
                    wh0 = mr["width_height"]
                    images = np.full((L, int(wh0[1]), int(wh0[0]), 3), 255, dtype=np.uint8)
                    Log.warn(f"[E9q] {vid}: video missing, rendering on white background")
                Krender = mr["K"]
                wh = mr["width_height"]
                scale = images.shape[2] / float(wh[0]) if hasattr(wh, "__len__") else 1.0
                rend = Renderer(int(wh[0]), int(wh[1]), device="cuda", faces=faces, K=Krender)
                fid = mr["frame_id"].cpu().numpy() if hasattr(mr["frame_id"], "cpu") else np.asarray(mr["frame_id"])
                rf = mr.get("resize_factor", 0.25)
                for t in picks:
                    t = int(t)
                    img = images[fid[t]] if t < len(fid) and fid[t] < len(images) else images[min(t, len(images) - 1)]
                    imageio.imwrite(seq_dir / f"frame_f{t}.png", img)
                    base = rend.render_mesh(gt_verts_c[t].cuda(), img, [0.72, 0.72, 0.68])
                    for name, color in [("control", [0.75, 0.35, 0.35]), ("casca", [0.30, 0.55, 0.80])]:
                        # root-align for visualization: what the MPJPE number measures
                        v_al = res[name]["verts"][t].cuda() - res[name]["pelvis"][t].cuda() + gt_pelvis[t]
                        ov = rend.render_mesh(v_al, base, color)
                        imageio.imwrite(seq_dir / f"overlay_{name}_f{t}.png", ov)
                Log.info(f"[E9q] {vid}: overlays at {list(map(int, picks))}")
            except Exception as e:
                Log.warn(f"[E9q] {vid}: overlay rendering skipped ({e}); curves/traj saved")
            with open(seq_dir / "picks.json", "w") as f:
                json.dump({"picks": [int(x) for x in picks],
                           "frame_mpjpe": {n: [float(res[n]["mpjpe"][int(t)]) for t in picks]
                                           for n in res},
                           "mean_ctrl": float(np.nanmean(res["control"]["mpjpe"])),
                           "mean_casca": float(np.nanmean(res["casca"]["mpjpe"]))}, f)
    Log.info("[E9q] done")


if __name__ == "__main__":
    register_store_gvhmr()
    main()
