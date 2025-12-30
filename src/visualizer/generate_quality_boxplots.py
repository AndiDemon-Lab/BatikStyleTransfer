"""
Generate box plots for quality metrics only (SSIM, PSNR, LPIPS).
Training time will be in separate visualization.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set publication-quality style
plt.style.use("seaborn-v0_8-paper")
sns.set_palette("husl")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 10
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["xtick.labelsize"] = 9
plt.rcParams["ytick.labelsize"] = 9
plt.rcParams["legend.fontsize"] = 9

# Create output directory
output_dir = Path("outputs/visualizations")
output_dir.mkdir(exist_ok=True)

# Load data
print("Loading data from outputs/full_analysis_results.xlsx...")
df = pd.read_excel("outputs/full_analysis_results.xlsx", sheet_name="Raw Data")
print(f"Loaded {len(df)} experiments")

# Define model order and colors
model_order = ["vgg16", "vgg19", "inception_v3", "resnet50", "resnet101"]
model_labels = {
    "vgg16": "VGG16",
    "vgg19": "VGG19",
    "inception_v3": "Inception V3",
    "resnet50": "ResNet50",
    "resnet101": "ResNet101",
}

# ============================================================================
# QUALITY METRICS BOX PLOTS (3 panels: SSIM, PSNR, LPIPS)
# ============================================================================
print("\nCreating quality metrics box plots...")

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle(
    "Quality Metrics Comparison Across CNN Architectures",
    fontsize=14,
    fontweight="bold",
)

metrics = [
    ("final_ssim", "SSIM (Structural Similarity)", "higher is better"),
    ("final_psnr", "PSNR (dB)", "higher is better"),
    ("final_lpips", "LPIPS (Perceptual Distance)", "lower is better"),
]

for idx, (metric, title, note) in enumerate(metrics):
    ax = axes[idx]

    # Prepare data
    plot_data = []
    for model in model_order:
        model_data = df[df["model_name"] == model][metric].dropna()
        plot_data.append(model_data)

    # Box plot
    bp = ax.boxplot(
        plot_data,
        labels=[model_labels[m] for m in model_order],
        patch_artist=True,
        showmeans=True,
        meanprops=dict(marker="D", markerfacecolor="red", markersize=5),
    )

    # Color boxes
    colors = sns.color_palette("husl", len(model_order))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_ylabel(title)
    ax.set_xlabel("Model Architecture")
    ax.grid(axis="y", alpha=0.3)
    ax.set_title(f"{title}\n({note})", fontsize=10)

    # Rotate x labels
    ax.tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.savefig(output_dir / "quality_metrics_boxplots.png", bbox_inches="tight")
print(f"  Saved: quality_metrics_boxplots.png")
plt.close()

# ============================================================================
# TRAINING TIME BOX PLOT (separate, single panel)
# ============================================================================
print("\nCreating training time box plot...")

fig, ax = plt.subplots(1, 1, figsize=(10, 6))
fig.suptitle(
    "Training Time Comparison Across CNN Architectures", fontsize=14, fontweight="bold"
)

# Prepare data
plot_data = []
for model in model_order:
    model_data = df[df["model_name"] == model]["training_time_seconds"].dropna()
    plot_data.append(model_data)

# Box plot
bp = ax.boxplot(
    plot_data,
    labels=[model_labels[m] for m in model_order],
    patch_artist=True,
    showmeans=True,
    meanprops=dict(marker="D", markerfacecolor="red", markersize=6),
)

# Color boxes
colors = sns.color_palette("husl", len(model_order))
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)

ax.set_ylabel("Training Time (seconds)", fontsize=12)
ax.set_xlabel("Model Architecture", fontsize=12)
ax.grid(axis="y", alpha=0.3)
ax.tick_params(axis="x", rotation=45)

# Add horizontal reference lines for key values
ax.axhline(
    y=25,
    color="green",
    linestyle="--",
    alpha=0.5,
    linewidth=1,
    label="ResNet avg (~25s)",
)
ax.axhline(
    y=150, color="red", linestyle="--", alpha=0.5, linewidth=1, label="VGG avg (~150s)"
)
ax.legend(loc="upper right")

plt.tight_layout()
plt.savefig(output_dir / "training_time_boxplot.png", bbox_inches="tight")
print(f"  Saved: training_time_boxplot.png")
plt.close()

print("\n" + "=" * 80)
print("QUALITY AND TRAINING TIME PLOTS GENERATED!")
print("=" * 80)
print(f"\nOutput directory: {output_dir.absolute()}")
print("\nGenerated files:")
print("  1. quality_metrics_boxplots.png - SSIM, PSNR, LPIPS only (3 panels)")
print("  2. training_time_boxplot.png - Training time only (1 panel)")
print("=" * 80)
