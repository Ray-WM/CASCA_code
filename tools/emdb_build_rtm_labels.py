"""EMDB detector swap, step 2 (runs on the GPU server): build emdb_vit_v4_rtm.pt.

Copies emdb_vit_v4.pt and replaces kp2d of every sequence with the
RTMPose detections uploaded to inputs/EMDB/rtm_kp2d/*.npz. Features,
bbx_xys, GT and camera data stay untouched.
"""
import torch, numpy as np
from pathlib import Path

SUP = Path("inputs/EMDB/hmr4d_support")
NPZ = Path("inputs/EMDB/rtm_kp2d")

labels = torch.load(SUP / "emdb_vit_v4.pt")
npzs = {p.stem: p for p in NPZ.glob("*.npz")}
replaced = 0
for vid, lab in labels.items():
    if vid not in npzs:
        print(f"WARN no rtm npz for {vid}, keeping ViTPose")
        continue
    kp = np.load(npzs[vid])["kp2d"]
    old = lab["kp2d"]
    assert tuple(kp.shape) == tuple(old.shape), (vid, kp.shape, old.shape)
    lab["kp2d"] = torch.from_numpy(kp).to(old.dtype)
    replaced += 1
torch.save(labels, SUP / "emdb_vit_v4_rtm.pt")
print(f"BUILD_OK replaced {replaced}/{len(labels)} vids -> {SUP/'emdb_vit_v4_rtm.pt'}")
