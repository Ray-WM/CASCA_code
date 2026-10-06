# CASCA: Confidence-Aware Selective Cross-Attention for Robust World-Grounded Human Motion Recovery

Code accompanying our IEEE TCSVT submission. CASCA encodes 2D-pose and
visual-context signals in two isolated temporal streams and re-couples
them through zero-initialized, reliability-conditioned cross-attention.
This repository contains the architecture, all seven controlled encoder
variants used in the paper's dissection, and the full audit protocol
(gate-blind probes, smoothing and capacity controls, unstructured and
structured corruption stress tests, detector swap).

## Acknowledgment

This codebase is built on **GVHMR** (Shen et al., "World-Grounded Human
Motion Recovery via Gravity-View Coordinates", SIGGRAPH Asia 2024,
https://github.com/zju3dv/GVHMR). We thank the authors for releasing
their code and models. All inherited code remains under the original
license (see `LICENSE`); the original documentation is kept as
`README_GVHMR.md`. Our additions are released for research purposes.

## Environment

Same as upstream GVHMR (see `README_GVHMR.md` and `docs/` for details):

```bash
conda create -n gvhmr python=3.10 -y && conda activate gvhmr
pip install -r requirements.txt
pip install -e .
```

PyTorch 2.x with CUDA is required. Dataset and preprocessed-input
preparation (`inputs/`) follows the upstream GVHMR instructions.

## What this repository adds

| Path | Content |
|---|---|
| `hmr4d/network/gvhmr/casca_transformer.py` | CASCA dual-stream encoder |
| `hmr4d/network/gvhmr/{fuze,shared_cond,gavphr,opti_crossAtt,residual_crossatt}_transformer.py` | Controlled variants (Fuze, SharedCond, legacy CrossAtt, ...) |
| `hmr4d/configs/exp/gvhmr/mixed/*.yaml`, `.../gavphr/*.yaml` | Experiment configs for all arms |
| `hmr4d/model/gvhmr/callbacks/e4r_export.py` | Per-frame prediction export for analysis |
| `tools/e1_smoothing.py` | Output-filtering control (Savitzky-Golay / OneEuro sweeps) |
| `tools/e3_probe.py`, `tools/e3_probe_multi.py` | Gate-blind spillover probes |
| `tools/e7_attention_decay.py` | Attention locality measurement |
| `tools/e8_stress.py`, `tools/e8_perseq.py` | Unstructured corruption stress test (+ per-sequence bootstrap) |
| `tools/e11_realistic_stress.py` | Structured realistic corruption (left-right swap / stuck / hijack) |
| `tools/e9_qualitative.py`, `tools/bench_encoder.py` | Qualitative renders; encoder-level efficiency benchmark |
| `tools/emdb_rtmpose_extract.py`, `tools/emdb_build_rtm_labels.py` | Detector-swap experiment (ViTPose to RTMPose) |
| `tools/e4r_analysis.py`, `tools/p1_review_analyses.py` | Incidence census, gate statistics, dose-response analyses |
| `tools/plot_*.py`, `tools/paper_figs/` | Figure generation from exported results |

## Running the main arms

Training (one 80GB GPU, or adjust batch size with gradient accumulation):

```bash
python tools/train.py exp=gvhmr/mixed/mixed          # GVHMR-control (baseline)
python tools/train.py exp=gvhmr/mixed/casca          # CASCA
python tools/train.py exp=gvhmr/mixed/control_l      # capacity control
python tools/train.py exp=gvhmr/mixed/shared_cond    # conditioning in shared encoder
python tools/train.py exp=gvhmr/gavphr/fuze          # isolation without exchange
python tools/train.py exp=gvhmr/gavphr/crossatt      # legacy cross-attention
ABLATION_UNCOND_EXCHANGE=1 python tools/train.py exp=gvhmr/mixed/casca  # Uncond minimal pair
```

Official evaluation with per-frame export:

```bash
python tools/train.py exp=gvhmr/mixed/casca global/task=gvhmr/e4r_export \
  ckpt_path=<checkpoint.ckpt>
```

Stress tests (checkpoint-only, no retraining):

```bash
python tools/e8_stress.py exp=gvhmr/mixed/casca ckpt_path=<ckpt> +e8_tag=casca
python tools/e11_realistic_stress.py exp=gvhmr/mixed/casca ckpt_path=<ckpt> +e11_tag=casca
```

Retrained checkpoints for all arms will be released upon acceptance.
