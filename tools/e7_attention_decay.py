"""E7: measure attention decay vs. temporal distance on a trained model
(empirical support for Assumption A / banded attention in the TCSVT paper).

Patches RoPEAttention to accumulate, per layer, the mean post-softmax
attention weight as a function of |t - s| (aggregated on the fly, so long
sequences never materialize more than one L x L map at a time).

Run on the REMOTE machine (1 GPU is enough):
  python tools/e7_attention_decay.py exp=<exp_of_model> ckpt_path=<ckpt> \
      [+e7_dataset=emdb1] [+e7_n_seqs=8] [+e7_tag=control]

Outputs: results/e7_attention/<tag>/decay.json  (per-layer + mean curves,
         fitted decay rate, 1/10 and 1/100 distances, hard bandwidth)
         results/e7_attention/<tag>/decay.pdf
"""

import json
import math
from collections import defaultdict
from pathlib import Path

import hydra
import numpy as np
import torch
from einops import einsum
from omegaconf import DictConfig

from hmr4d.configs import register_store_gvhmr
from hmr4d.utils.pylogger import Log

DMAX = 150  # max temporal distance to record


# ---------------- attention capture ---------------- #
ACCUM = defaultdict(lambda: [np.zeros(DMAX + 1), np.zeros(DMAX + 1)])  # name -> [sum, cnt]


def make_patched_forward(orig_cls):
    def patched(self, x, attn_mask=None, key_padding_mask=None):
        B, L, _ = x.shape
        xq, xk, xv = self.query(x), self.key(x), self.value(x)
        xq = xq.reshape(B, L, self.num_heads, -1).transpose(1, 2)
        xk = xk.reshape(B, L, self.num_heads, -1).transpose(1, 2)
        xv = xv.reshape(B, L, self.num_heads, -1).transpose(1, 2)
        xq = self.rope.rotate_queries_or_keys(xq)
        xk = self.rope.rotate_queries_or_keys(xk)
        attn_score = einsum(xq, xk, "b n i c, b n j c -> b n i j") / math.sqrt(self.head_dim)
        if attn_mask is not None:
            am = attn_mask.reshape(1, 1, L, L).expand(B, self.num_heads, -1, -1)
            attn_score = attn_score.masked_fill(am, float("-inf"))
        if key_padding_mask is not None:
            kpm = key_padding_mask.reshape(B, 1, 1, L).expand(-1, self.num_heads, L, -1)
            attn_score = attn_score.masked_fill(kpm, float("-inf"))
        attn = torch.softmax(attn_score, dim=-1)

        # --- accumulate mean attention vs temporal distance (both directions) ---
        name = getattr(self, "_e7_name", "unnamed")
        s, c = ACCUM[name]
        head_mean = attn.mean(dim=1)  # (B, L, L)
        for d in range(0, min(DMAX, L - 1) + 1):
            diag_u = torch.diagonal(head_mean, offset=d, dim1=-2, dim2=-1)
            vals = diag_u
            if d > 0:
                diag_l = torch.diagonal(head_mean, offset=-d, dim1=-2, dim2=-1)
                vals = torch.cat([diag_u, diag_l], dim=-1)
            finite = vals[torch.isfinite(vals)]
            if finite.numel():
                s[d] += finite.sum().item()
                c[d] += finite.numel()

        attn = self.dropout(attn)
        out = einsum(attn, xv, "b n i j, b n j c -> b n i c")
        out = out.transpose(1, 2).reshape(B, L, -1)
        return self.proj(out)

    return patched


# ---------------- main ---------------- #
@hydra.main(version_base="1.3", config_path="../hmr4d/configs", config_name="train")
def main(cfg: DictConfig) -> None:
    from hmr4d.utils.geo.hmr_cam import normalize_kp2d
    from hmr4d.utils.net_utils import load_pretrained_model
    import hmr4d.network.base_arch.transformer.encoder_rope as er

    tag = cfg.get("e7_tag", "model")
    ds_name = cfg.get("e7_dataset", "emdb1")
    n_seqs = int(cfg.get("e7_n_seqs", 8))
    out_dir = Path(f"results/e7_attention/{tag}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # model
    model = hydra.utils.instantiate(cfg.model, _recursive_=False)
    assert cfg.ckpt_path is not None, "pass ckpt_path=..."
    load_pretrained_model(model, cfg.ckpt_path)
    model = model.cuda().eval()

    # name the attention modules, then patch the class
    n_attn = 0
    for name, mod in model.named_modules():
        if isinstance(mod, er.RoPEAttention):
            mod._e7_name = name
            n_attn += 1
    Log.info(f"[E7] found {n_attn} RoPEAttention modules")
    er.RoPEAttention.forward = make_patched_forward(er.RoPEAttention)

    # dataset
    if ds_name == "emdb1":
        from hmr4d.dataset.emdb.emdb_motion_test import EmdbSmplFullSeqDataset

        ds = EmdbSmplFullSeqDataset(split=1)
    elif ds_name == "rich":
        from hmr4d.dataset.rich.rich_motion_test import RichSmplFullSeqDataset

        ds = RichSmplFullSeqDataset()
    else:
        raise ValueError(ds_name)

    hard_bandwidth = None
    with torch.no_grad():
        for i in range(min(n_seqs, len(ds))):
            d = ds[i]
            batch = {}
            for k in ["kp2d", "bbx_xys", "K_fullimg", "cam_angvel", "f_imgseq"]:
                batch[k] = d[k][None].cuda().float()
            length = torch.tensor([d["length"]]).cuda()
            obs = normalize_kp2d(batch["kp2d"], batch["bbx_xys"])
            if "mask" in d:
                obs[0, ~d["mask"].bool().cuda()] = 0
            batch_ = {
                "length": length,
                "obs": obs,
                "bbx_xys": batch["bbx_xys"],
                "K_fullimg": batch["K_fullimg"],
                "cam_angvel": batch["cam_angvel"],
                "f_imgseq": batch["f_imgseq"],
            }
            model.pipeline.forward(batch_, train=False, postproc=False)
            if hard_bandwidth is None:
                net = model.pipeline.denoiser3d
                hard_bandwidth = int(getattr(net, "max_len", -1))
            Log.info(f"[E7] {i+1}/{n_seqs}: {d['meta']['vid']} (L={d['length']})")

    # ---- summarize ----
    layers = {}
    for name, (s, c) in ACCUM.items():
        with np.errstate(invalid="ignore", divide="ignore"):
            layers[name] = np.where(c > 0, s / np.maximum(c, 1), np.nan)
    mean_curve = np.nanmean(np.stack(list(layers.values())), axis=0)

    # exponential fit on distances 1..60: ln y = c0 + c1 * d
    dmax_fit = 60
    dists = np.arange(1, dmax_fit + 1)
    y = mean_curve[1 : dmax_fit + 1]
    ok = np.isfinite(y) & (y > 0)
    c1, c0 = np.polyfit(dists[ok], np.log(y[ok]), 1)
    beta_hat = float(np.exp(c1))
    peak = mean_curve[1]

    def dist_to_frac(frac):
        idx = np.where(mean_curve[1:] <= peak * frac)[0]
        return int(idx[0] + 1) if len(idx) else None

    summary = {
        "tag": tag,
        "dataset": ds_name,
        "n_seqs": n_seqs,
        "hard_bandwidth_max_len": hard_bandwidth,
        "beta_hat_per_frame": beta_hat,
        "fit_range": [1, dmax_fit],
        "dist_to_1_10": dist_to_frac(0.1),
        "dist_to_1_100": dist_to_frac(0.01),
        "mean_curve": mean_curve.tolist(),
        "layers": {k: v.tolist() for k, v in layers.items()},
    }
    with open(out_dir / "decay.json", "w") as f:
        json.dump(summary, f, indent=2)
    Log.info(
        f"[E7] beta_hat={beta_hat:.4f}/frame, 1/10 at d={summary['dist_to_1_10']}, "
        f"1/100 at d={summary['dist_to_1_100']}, hard bandwidth={hard_bandwidth}"
    )

    # ---- plot ----
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    from paper_style import W_COL, setup

    import matplotlib.pyplot as plt

    setup()
    fig, ax = plt.subplots(figsize=(W_COL, 2.3))
    for name, v in layers.items():
        ax.plot(np.arange(len(v)), v, color="#0072B2", alpha=0.18, lw=0.7)
    ax.plot(np.arange(len(mean_curve)), mean_curve, color="#0072B2", lw=1.6, label="layer mean")
    fit_y = np.exp(c0 + c1 * np.arange(1, DMAX + 1))
    ax.plot(np.arange(1, DMAX + 1), fit_y, color="#555555", ls="--", lw=1.0,
            label=rf"fit $\hat\beta$={beta_hat:.3f}/frame")
    ax.set_yscale("log")
    ax.set_xlabel("temporal distance |t-s| (frames)")
    ax.set_ylabel("mean attention weight")
    ax.legend()
    fig.savefig(out_dir / "decay.pdf")
    Log.info(f"[E7] wrote {out_dir / 'decay.pdf'}")


if __name__ == "__main__":
    register_store_gvhmr()
    main()
