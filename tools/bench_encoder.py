"""Benchmark encoder-level inference time / peak memory on a synthetic
1000-frame sequence (fills the efficiency-table gaps after internal review).

Run on the REMOTE machine (1 GPU, run each model separately for clean peaks):
  python tools/bench_encoder.py exp=gvhmr/mixed/mixed ckpt_path=<control.ckpt> +bench_tag=control
  python tools/bench_encoder.py exp=<control_l exp>   ckpt_path=<ctrl_l.ckpt>  +bench_tag=control_l
  python tools/bench_encoder.py exp=<casca exp>       ckpt_path=<casca.ckpt>   +bench_tag=casca

Output: results/bench/<tag>.json  {params_M, encoder_ms_mean, encoder_ms_std, peak_mem_MB}
"""

import json
import time
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

from hmr4d.configs import register_store_gvhmr
from hmr4d.utils.pylogger import Log

L = 1000
N_WARMUP = 5
N_TIMED = 20


@hydra.main(version_base="1.3", config_path="../hmr4d/configs", config_name="train")
def main(cfg: DictConfig) -> None:
    from hmr4d.utils.net_utils import load_pretrained_model

    tag = cfg.get("bench_tag", "model")
    out_dir = Path("results/bench")
    out_dir.mkdir(parents=True, exist_ok=True)

    model = hydra.utils.instantiate(cfg.model, _recursive_=False)
    assert cfg.ckpt_path is not None
    load_pretrained_model(model, cfg.ckpt_path)
    model = model.cuda().eval()
    n_params = sum(p.numel() for p in model.pipeline.parameters()) / 1e6

    batch = dict(
        length=torch.tensor([L]).cuda(),
        obs=torch.rand(1, L, 17, 3).cuda(),
        bbx_xys=torch.tensor([[128.0, 128.0, 200.0]]).repeat(1, L, 1).cuda(),
        K_fullimg=torch.eye(3)[None, None].repeat(1, L, 1, 1).cuda() * 1000,
        cam_angvel=torch.zeros(1, L, 6).cuda(),
        f_imgseq=torch.rand(1, L, 1024).cuda(),
    )
    batch["K_fullimg"][..., 2, 2] = 1.0

    with torch.no_grad():
        for _ in range(N_WARMUP):
            model.pipeline.forward(batch, train=False, postproc=False)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        times = []
        for _ in range(N_TIMED):
            t0 = time.perf_counter()
            model.pipeline.forward(batch, train=False, postproc=False)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000.0)
    times = torch.tensor(times)
    peak = torch.cuda.max_memory_allocated() / 1024 / 1024

    res = {"tag": tag, "params_M": round(n_params, 2), "L": L,
           "encoder_ms_mean": round(times.mean().item(), 2),
           "encoder_ms_std": round(times.std().item(), 2),
           "peak_mem_MB": round(peak, 1)}
    json.dump(res, open(out_dir / f"{tag}.json", "w"), indent=1)
    Log.info(f"[bench] {res}")


if __name__ == "__main__":
    register_store_gvhmr()
    main()
