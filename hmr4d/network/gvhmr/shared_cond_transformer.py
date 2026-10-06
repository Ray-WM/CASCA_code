"""P3.1 (post-internal-review): the missing design-space cell —
SHARED-token encoder + reliability CONDITIONING, no stream isolation.

The single-encoder baseline (NetworkEncoderRoPE) is kept intact; the only
change is that the pose contribution to the shared token is down-weighted
by a learned, reliability-conditioned per-frame gate:

    x_t = w_t * phi_pose(u_t) + phi_cam + phi_ang + phi_img,
    w_t = 1 - alpha * d_t,   alpha zero-initialized (starts as exact baseline)

d_t comes from the SAME 18-D cue vector as CASCA (17 joint confidences +
camera angular-velocity magnitude, shared DifficultyEstimator MLP). This is
the classical dynamic-stream-weighting move (AVSR-style) inside a shared
token: conditioning without isolation. If gate-blind corruption still
propagates here (it is invisible to d_t by construction), isolation — not
conditioning — is what containment requires.

Warm-start: architecture is a strict superset of the baseline, so
load_baseline_weights() loads every baseline parameter; only the
DifficultyEstimator and alpha are fresh (and alpha=0 keeps the start
functionally identical to the loaded baseline).
"""

import torch
import torch.nn as nn

from hmr4d.configs import MainStore, builds
from hmr4d.network.gvhmr.relative_transformer import NetworkEncoderRoPE
from hmr4d.network.gvhmr.casca_transformer import DifficultyEstimator
from hmr4d.utils.pylogger import Log


class NetworkEncoderRoPESharedCond(NetworkEncoderRoPE):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.difficulty_estimator = DifficultyEstimator(self.cam_angvel_dim)
        self.cond_alpha = nn.Parameter(torch.zeros(1))

    def forward(self, length, obs=None, f_cliffcam=None, f_cam_angvel=None, f_imgseq=None):
        B, L, J, C = obs.shape
        assert J == 17 and C == 3

        # Reliability gate from raw cues (same inputs as CASCA's estimator)
        joint_conf = obs[..., 2]  # (B, L, 17)
        d = self.difficulty_estimator(joint_conf, f_cam_angvel)  # (B, L, 1)
        w = 1.0 - self.cond_alpha * d  # (B, L, 1); alpha=0 -> exact baseline
        self._last_d = d.detach()
        self._last_w = w.detach()

        # ----- identical to baseline except the w-scaling of the pose token -----
        obs = obs.clone()
        visible_mask = obs[..., [2]] > 0.5
        obs[~visible_mask[..., 0]] = 0
        f_obs = self.learned_pos_linear(obs[..., :2])
        f_obs = f_obs * visible_mask + self.learned_pos_params.repeat(B, L, 1, 1) * ~visible_mask
        x = self.embed_noisyobs(f_obs.view(B, L, -1)) * w  # <-- conditioned pose weight

        f_to_add = [self.cliffcam_embedder(f_cliffcam)]
        if hasattr(self, "cam_angvel_embedder"):
            f_to_add.append(self.cam_angvel_embedder(f_cam_angvel))
        if f_imgseq is not None and hasattr(self, "imgseq_embedder"):
            f_to_add.append(self.imgseq_embedder(f_imgseq))
        for f_delta in f_to_add:
            x = x + f_delta

        return self._encode_and_output(x, length, L)

    def _encode_and_output(self, x, length, L):
        """Baseline tail (mask -> blocks -> heads), factored for reuse."""
        from einops import repeat

        from hmr4d.utils.net_utils import length_to_mask

        B = x.shape[0]
        assert B == length.size(0)
        pmask = ~length_to_mask(length, L)

        if L > self.max_len:
            attnmask = torch.ones((L, L), device=x.device, dtype=torch.bool)
            for i in range(L):
                min_ind = max(0, i - self.max_len // 2)
                max_ind = min(L, i + self.max_len // 2)
                max_ind = max(self.max_len, max_ind)
                min_ind = min(L - self.max_len, min_ind)
                attnmask[i, min_ind:max_ind] = False
        else:
            attnmask = None

        for block in self.blocks:
            x = block(x, attn_mask=attnmask, tgt_key_padding_mask=pmask)

        sample = self.final_layer(x)
        if self.avgbeta:
            betas = (sample[..., 126:136] * (~pmask[..., None])).sum(1) / length[:, None]
            betas = repeat(betas, "b c -> b l c", l=L)
            sample = torch.cat([sample[..., :126], betas, sample[..., 136:]], dim=-1)

        pred_cam = None
        if self.pred_cam_head:
            pred_cam = self.pred_cam_head(x)
            pred_cam = pred_cam * self.pred_cam_std + self.pred_cam_mean
            torch.clamp_min_(pred_cam[..., 0], 0.25)

        static_conf_logits = None
        if self.static_conf_head:
            static_conf_logits = self.static_conf_head(x)

        return {
            "pred_context": x,
            "pred_x": sample,
            "pred_cam": pred_cam,
            "static_conf_logits": static_conf_logits,
        }

    def load_baseline_weights(self, ckpt_path):
        """Load ALL baseline weights (strict superset); estimator/alpha stay fresh."""
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        full_sd = checkpoint.get("state_dict", checkpoint)
        prefix = None
        for key in full_sd.keys():
            if "learned_pos_linear" in key:
                prefix = key.split("learned_pos_linear")[0]
                break
        assert prefix is not None, "cannot locate encoder prefix in checkpoint"
        sd = {k[len(prefix):]: v for k, v in full_sd.items() if k.startswith(prefix)}
        missing, unexpected = self.load_state_dict(sd, strict=False)
        fresh = [k for k in missing if k.startswith(("difficulty_estimator", "cond_alpha"))]
        assert len(missing) == len(fresh), f"unexpected missing keys: {set(missing) - set(fresh)}"
        assert not unexpected, f"unexpected keys: {unexpected}"
        Log.info(f"[SharedCond] warm-started {len(sd) - 0} baseline tensors; "
                 f"fresh: {len(fresh)} (difficulty_estimator + cond_alpha)")


MainStore.store(
    name="shared_cond_transformer",
    node=builds(NetworkEncoderRoPESharedCond, populate_full_signature=True),
    group="network/gvhmr",
)
