# save as tools/plot_e1.py
import json
import numpy as np
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

save_dir = "results/e1_noise"

def compute_spillover(clean_file, noisy_file, max_dist=20):
    with open(f"{save_dir}/{clean_file}") as f:
        clean = json.load(f)
    with open(f"{save_dir}/{noisy_file}") as f:
        noisy = json.load(f)
    
    delta_by_dist = defaultdict(list)
    for vid in noisy:
        if vid not in clean:
            continue
        c = np.array(clean[vid]["mpjpe"])
        n = np.array(noisy[vid]["mpjpe"])
        ns = noisy[vid]["noise_start"]
        ne = noisy[vid]["noise_end"]
        center = (ns + ne) / 2.0
        L = min(len(c), len(n))
        for t in range(L):
            d = int(round(t - center))
            if abs(d) <= max_dist:
                delta_by_dist[d].append(n[t] - c[t])
    
    dists = sorted(delta_by_dist.keys())
    means = [np.mean(delta_by_dist[d]) for d in dists]
    stds = [np.std(delta_by_dist[d]) for d in dists]
    return dists, np.array(means), np.array(stds)

# 计算两条曲线
d_b, m_b, s_b = compute_spillover("baseline_clean_rich.json", "baseline_noisy_rich.json")
d_c, m_c, s_c = compute_spillover("casca_clean_rich.json", "casca_noisy_rich.json")

# 画图
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(d_b, m_b, "r-", linewidth=2.5, label="GVHMR (Single Encoder)")
ax.fill_between(d_b, m_b - s_b, m_b + s_b, color="red", alpha=0.12)

ax.plot(d_c, m_c, "b-", linewidth=2.5, label="CASCA (Dual-Stream, Ours)")
ax.fill_between(d_c, m_c - s_c, m_c + s_c, color="blue", alpha=0.12)

ax.axvline(0, color="gray", ls="--", alpha=0.5)
ax.axvspan(-2.5, 2.5, color="gray", alpha=0.08, label="Noise region (5 frames)")

ax.set_xlabel("Distance from Noise Center (frames)", fontsize=14)
ax.set_ylabel("Δ MPJPE (mm)", fontsize=14)
ax.set_title("Temporal Noise Propagation: Single Encoder vs Dual-Stream", fontsize=15)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{save_dir}/noise_spillover_curve.png", dpi=150)
plt.savefig(f"{save_dir}/noise_spillover_curve.pdf")
print("Saved: noise_spillover_curve.png/pdf")

# 打印关键数字
print(f"\n=== Key Statistics ===")
# noise区域之外(-20到-5和+5到+20)的平均delta
outer_b = [m_b[i] for i, d in enumerate(d_b) if 5 <= abs(d) <= 20]
outer_c = [m_c[i] for i, d in enumerate(d_c) if 5 <= abs(d) <= 20]
print(f"Baseline - avg spillover outside noise region: {np.mean(outer_b):.2f} mm")
print(f"CASCA    - avg spillover outside noise region: {np.mean(outer_c):.2f} mm")
print(f"Reduction: {(1 - np.mean(outer_c)/np.mean(outer_b))*100:.1f}%")
