# save as tools/plot_e3.py
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

save_dir = "results/e3_difficulty"
with open(f"{save_dir}/rich_difficulty.json") as f:
    data = json.load(f)

difficulty = data["difficulty"]
confidence = data["confidence"]

# ===== 图1: 选几个代表性序列画时间轴 =====
# 选difficulty变化最大的序列
scored = [(np.std(difficulty[vid]), vid) for vid in difficulty if vid in confidence]
scored.sort(reverse=True)

for rank, (std, vid) in enumerate(scored[:5]):
    diff = np.array(difficulty[vid])
    conf = np.array(confidence[vid]["mean_conf"])
    occ = np.array(confidence[vid]["num_low_conf"])
    L = len(diff)
    frames = np.arange(L)
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    
    axes[0].plot(frames, diff, "r-", linewidth=1.5)
    axes[0].fill_between(frames, 0, diff, color="red", alpha=0.15)
    axes[0].set_ylabel("Difficulty Score")
    axes[0].set_ylim(0, 1)
    axes[0].set_title(f"{vid} (std={std:.4f})")
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(frames, conf, "g-", linewidth=1.5)
    axes[1].set_ylabel("Mean Joint Conf")
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, alpha=0.3)
    
    axes[2].bar(frames, occ, color="orange", alpha=0.7, width=1.0)
    axes[2].set_ylabel("# Occluded Joints")
    axes[2].set_ylim(0, 17)
    axes[2].set_xlabel("Frame")
    axes[2].grid(True, alpha=0.3)
    
    # 标记遮挡区域
    occ_mask = occ >= 3
    for ax in axes:
        in_range = False
        start = 0
        for i in range(len(occ_mask)):
            if occ_mask[i] and not in_range:
                start = i; in_range = True
            elif not occ_mask[i] and in_range:
                for ax2 in axes: ax2.axvspan(start, i, color="yellow", alpha=0.15)
                in_range = False
        if in_range:
            for ax2 in axes: ax2.axvspan(start, len(occ_mask), color="yellow", alpha=0.15)
    
    plt.tight_layout()
    safe_vid = vid.replace("/", "_")
    plt.savefig(f"{save_dir}/seq_{rank}_{safe_vid}.png", dpi=120)
    plt.close()
    print(f"Saved: seq_{rank}_{safe_vid}.png")

# ===== 图2: 所有帧的difficulty分布 (clean vs occluded) =====
all_diff_clean = []
all_diff_occ = []

for vid in difficulty:
    if vid not in confidence:
        continue
    diff = np.array(difficulty[vid])
    occ = np.array(confidence[vid]["num_low_conf"])
    all_diff_clean.extend(diff[occ < 3].tolist())
    all_diff_occ.extend(diff[occ >= 3].tolist())

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(all_diff_clean, bins=50, alpha=0.6, color="green", label=f"Clean frames (n={len(all_diff_clean)})", density=True)
ax.hist(all_diff_occ, bins=50, alpha=0.6, color="red", label=f"Occluded frames (n={len(all_diff_occ)})", density=True)
ax.set_xlabel("Difficulty Score")
ax.set_ylabel("Density")
ax.set_title("CASCA Difficulty Distribution: Clean vs Occluded Frames")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{save_dir}/difficulty_distribution.png", dpi=150)
print("Saved: difficulty_distribution.png")