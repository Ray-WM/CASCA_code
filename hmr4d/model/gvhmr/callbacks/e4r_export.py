"""E4-R per-frame export for the dose-response analysis (TCSVT revision).

For every test sequence of RICH / EMDB-1 / EMDB-2 / 3DPW, export per-frame:
  - mpjpe        : dataset-standard camera-space MPJPE (mm), mirroring the
                   conventions of metric_rich / metric_emdb (24-joint SMPL,
                   pelvis=[1,2]) and metric_3dpw (14-joint, pelvis=[2,3])
  - mpjpe24      : unified 24-joint pelvis=[1,2] MPJPE (mm) for cross-dataset
                   comparability (differs from `mpjpe` only on 3DPW)
  - n_lowconf    : #joints with detection confidence < 0.5
  - mean_conf    : mean detection confidence over 17 joints
  - g_pix        : mean 2D reprojection error (pixels) of detected keypoints
                   vs. GT SMPL joints projected via K_fullimg (COCO17)
  - g_norm       : g_pix normalized by bbx size, scaled to a 256px reference
  - valid        : dataset valid mask (RICH: all True)
  - d_t, gate/update stats : only for models exposing them (CASCA)

Usage (remote):
  python tools/train.py exp=<exp_used_for_this_model> \
      global/task=gvhmr/e4r_export ckpt_path=<ckpt> \
      callbacks.e4r_export.tag=<casca|control|control_l|fuze|crossatt>

Outputs: results/e4r/<tag>/<dataset>.npz  (flat per-frame arrays)
"""

from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import pytorch_lightning as pl
from einops import einsum

from hmr4d.configs import MainStore, builds
from hmr4d.utils.comm.gather import all_gather
from hmr4d.utils.eval.eval_utils import batch_align_by_pelvis, compute_jpe
from hmr4d.utils.geo.hmr_cam import perspective_projection
from hmr4d.utils.geo_transform import apply_T_on_points
from hmr4d.utils.pylogger import Log
from hmr4d.utils.smplx_utils import make_smplx

DATASET_IDS = ("RICH", "EMDB_1", "EMDB_2", "3DPW")


class E4RExporter(pl.Callback):
    def __init__(self, out_dir="results/e4r", tag="model"):
        super().__init__()
        self.out_dir = Path(out_dir) / tag
        self.tag = tag

        # Body models, mirroring the metric callbacks' conventions
        self.smplx_rich = {
            g: make_smplx("rich-smplx", gender=g) for g in ["male", "female", "neutral"]
        }
        self.smpl_gendered = {"male": make_smplx("smpl", gender="male"), "female": make_smplx("smpl", gender="female")}
        self.smplx_pred = make_smplx("supermotion")  # EMDB / RICH-style pred decoding uses rich-smplx neutral
        self.smplx_pred_3dpw = make_smplx("supermotion_EVAL3DPW")

        self.J24 = torch.load("hmr4d/utils/body_model/smpl_neutral_J_regressor.pt")
        self.J14 = torch.load("hmr4d/utils/body_model/smpl_3dpw14_J_regressor_sparse.pt").to_dense()
        self.Jcoco17 = torch.load("hmr4d/utils/body_model/smpl_coco17_J_regressor.pt")
        self.smplx2smpl = torch.load("hmr4d/utils/body_model/smplx2smpl_sparse.pt")

        # dataset_id -> column -> list of np arrays (one per sequence)
        self.rows = {d: defaultdict(list) for d in DATASET_IDS}

        self.on_test_batch_end = self.on_validation_batch_end = self.on_predict_batch_end
        self.on_test_epoch_end = self.on_validation_epoch_end = self.on_predict_epoch_end

    # ---------------- helpers ---------------- #
    def _cuda(self):
        for g in self.smplx_rich:
            self.smplx_rich[g] = self.smplx_rich[g].cuda()
        for g in self.smpl_gendered:
            self.smpl_gendered[g] = self.smpl_gendered[g].cuda()
        self.smplx_pred = self.smplx_pred.cuda()
        self.smplx_pred_3dpw = self.smplx_pred_3dpw.cuda()
        self.J24 = self.J24.cuda()
        self.J14 = self.J14.cuda()
        self.Jcoco17 = self.Jcoco17.cuda()
        self.smplx2smpl = self.smplx2smpl.cuda()

    def _gt_c_verts(self, batch, dataset_id):
        """GT SMPL vertices in camera coordinates, (L, 6890, 3)."""
        gender = batch["gender"][0]
        if dataset_id == "RICH":
            params = {k: v[0] for k, v in batch["gt_smpl_params"].items()}
            w_out = self.smplx_rich[gender](**params)
            w_verts = torch.stack([torch.matmul(self.smplx2smpl, v_) for v_ in w_out.vertices])
        else:  # EMDB_1 / EMDB_2 / 3DPW: SMPL gendered, world frame
            params = {k: v[0] for k, v in batch["smpl_params"].items()}
            w_verts = self.smpl_gendered[gender](**params).vertices
        return apply_T_on_points(w_verts, batch["T_w2c"][0])

    def _pred_c_verts(self, outputs, dataset_id):
        params = outputs["pred_smpl_params_incam"]
        if dataset_id == "RICH":
            smpl_out = self.smplx_rich["neutral"](**params)
        elif dataset_id == "3DPW":
            smpl_out = self.smplx_pred_3dpw(**params)
        else:
            smpl_out = self.smplx_pred(**params)
        return torch.stack([torch.matmul(self.smplx2smpl, v_) for v_ in smpl_out.vertices])

    @staticmethod
    def _perframe_mpjpe(pred_j3d, target_j3d, pelvis_idxs):
        """Root-aligned per-frame MPJPE in mm, full timeline (no masking)."""
        pred, target = pred_j3d.cpu(), target_j3d.cpu()
        pred, target, _, _ = batch_align_by_pelvis([pred, target, pred.clone(), target.clone()], pelvis_idxs)
        return (compute_jpe(pred, target) * 1000.0).astype(np.float64)

    # ---------------- per-batch ---------------- #
    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        assert batch["B"] == 1
        dataset_id = batch["meta"][0]["dataset_id"]
        if dataset_id not in DATASET_IDS:
            return
        self._cuda()

        vid = batch["meta"][0]["vid"]
        L = batch["length"][0].item()
        cols = self.rows[dataset_id]

        # --- GT / pred joints (camera space) ---
        target_c_verts = self._gt_c_verts(batch, dataset_id)  # (L, V, 3)
        pred_c_verts = self._pred_c_verts(outputs, dataset_id)
        t_j24 = einsum(self.J24, target_c_verts, "j v, l v i -> l j i")
        p_j24 = einsum(self.J24, pred_c_verts, "j v, l v i -> l j i")
        mpjpe24 = self._perframe_mpjpe(p_j24, t_j24, pelvis_idxs=[1, 2])
        if dataset_id == "3DPW":
            t_j14 = einsum(self.J14, target_c_verts, "j v, l v i -> l j i")
            p_j14 = einsum(self.J14, pred_c_verts, "j v, l v i -> l j i")
            mpjpe = self._perframe_mpjpe(p_j14, t_j14, pelvis_idxs=[2, 3])
        else:
            mpjpe = mpjpe24

        # --- observation-degradation measures ---
        kp2d = batch["kp2d"][0]  # (L, 17, 3) full-image pixels + conf
        conf = kp2d[..., 2]
        n_lowconf = (conf < 0.5).sum(-1)
        mean_conf = conf.mean(-1)

        t_j17 = einsum(self.Jcoco17, target_c_verts, "j v, l v i -> l j i")  # (L, 17, 3)
        gt_2d = perspective_projection(t_j17[None], batch["K_fullimg"])[0]  # (L, 17, 2)
        in_front = t_j17[..., 2] > 0.3  # exclude degenerate projections
        err_2d = (kp2d[..., :2] - gt_2d).norm(dim=-1)  # (L, 17)
        g_pix = err_2d.masked_fill(~in_front, float("nan")).nanmean(-1)  # (L,)
        bbx_size = batch["bbx_xys"][0][..., 2].clamp(min=1.0)  # (L,)
        g_norm = g_pix / bbx_size * 256.0

        # --- valid mask ---
        if "mask" in batch and dataset_id != "RICH":
            valid = batch["mask"][0].bool()
        else:
            valid = torch.ones(L, dtype=torch.bool, device=kp2d.device)

        # --- gate signals (CASCA only; absent for other encoders) ---
        model_output = outputs.get("model_output", {})
        difficulty = model_output.get("difficulty", None)
        if difficulty is not None:
            d_t = difficulty[0, :, 0].detach()
        else:
            d_t = torch.full((L,), float("nan"), device=kp2d.device)
        gate_info = model_output.get("gate_info", {}) or {}
        pose_upd, vis_upd = [], []
        for k, v in gate_info.items():
            if k.endswith("pose_update_norm"):
                pose_upd.append(v[0, :, 0])
            elif k.endswith("vis_update_norm"):
                vis_upd.append(v[0, :, 0])
        pose_upd = torch.stack(pose_upd).mean(0) if pose_upd else torch.full((L,), float("nan"), device=kp2d.device)
        vis_upd = torch.stack(vis_upd).mean(0) if vis_upd else torch.full((L,), float("nan"), device=kp2d.device)

        # --- store ---
        def npf(x):
            return x.detach().cpu().numpy().astype(np.float64)

        cols["vid"].append(np.array([vid] * L))
        cols["frame"].append(np.arange(L, dtype=np.int64))
        cols["valid"].append(valid.cpu().numpy())
        cols["mpjpe"].append(np.asarray(mpjpe, dtype=np.float64))
        cols["mpjpe24"].append(np.asarray(mpjpe24, dtype=np.float64))
        cols["n_lowconf"].append(npf(n_lowconf))
        cols["mean_conf"].append(npf(mean_conf))
        cols["g_pix"].append(npf(g_pix))
        cols["g_norm"].append(npf(g_norm))
        cols["d_t"].append(npf(d_t))
        cols["pose_update_norm"].append(npf(pose_upd))
        cols["vis_update_norm"].append(npf(vis_upd))

    # ---------------- epoch end ---------------- #
    def on_predict_epoch_end(self, trainer, pl_module):
        # Concatenate local rows, then gather across DDP ranks (dedupe by vid)
        local = {}
        for d in DATASET_IDS:
            if not self.rows[d]["vid"]:
                continue
            local[d] = {k: np.concatenate(v) for k, v in self.rows[d].items()}

        with torch.inference_mode(False):
            gathered = all_gather(local)

        if trainer.local_rank != 0:
            self.rows = {d: defaultdict(list) for d in DATASET_IDS}
            return

        self.out_dir.mkdir(parents=True, exist_ok=True)
        for d in DATASET_IDS:
            parts = [g[d] for g in gathered if d in g]
            if not parts:
                continue
            merged = {k: np.concatenate([p[k] for p in parts]) for k in parts[0].keys()}
            # dedupe duplicated sequences (DistributedSampler padding): each
            # sequence is a contiguous block with frame = 0..L-1, so keep the
            # first block per vid.
            keep = np.zeros(len(merged["vid"]), dtype=bool)
            for v in np.unique(merged["vid"]):
                sel = np.where(merged["vid"] == v)[0]
                seq_len = int(merged["frame"][sel].max()) + 1
                keep[sel[:seq_len]] = True
            merged = {k: v[keep] for k, v in merged.items()}

            fn = self.out_dir / f"{d}.npz"
            np.savez_compressed(fn, **merged)
            n_seq = len(np.unique(merged["vid"]))
            Log.info(f"[E4R] {self.tag}/{d}: {n_seq} seqs, {len(merged['vid'])} frames -> {fn}")

        self.rows = {d: defaultdict(list) for d in DATASET_IDS}


MainStore.store(
    name="e4r_export",
    node=builds(E4RExporter, populate_full_signature=True),
    group="callbacks",
    package="callbacks.e4r_export",
)
