import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import json

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

# Load data from Excel
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
# 1. BOX PLOTS - Metrics Comparison
# ============================================================================
print("\n1. Creating box plots...")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle(
    "Quantitative Metrics Comparison Across CNN Architectures",
    fontsize=14,
    fontweight="bold",
)

metrics = [
    ("final_ssim", "SSIM (Structural Similarity)", "higher is better"),
    ("final_psnr", "PSNR (dB)", "higher is better"),
    ("final_lpips", "LPIPS (Perceptual Distance)", "lower is better"),
    ("training_time_seconds", "Training Time (seconds)", "lower is better"),
]

for idx, (metric, title, note) in enumerate(metrics):
    ax = axes[idx // 2, idx % 2]

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
plt.savefig(output_dir / "metrics_boxplots.png", bbox_inches="tight")
print(f"  Saved: metrics_boxplots.png")
plt.close()

# ============================================================================
# 2. BAR CHARTS - Mean Comparison with Error Bars
# ============================================================================
print("\n2. Creating bar charts...")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(
    "Mean Performance Metrics by Model Architecture", fontsize=14, fontweight="bold"
)

bar_metrics = [
    ("final_ssim", "SSIM ↑"),
    ("final_psnr", "PSNR (dB) ↑"),
    ("final_lpips", "LPIPS ↓"),
]

for idx, (metric, ylabel) in enumerate(bar_metrics):
    ax = axes[idx]

    means = []
    stds = []
    for model in model_order:
        model_data = df[df["model_name"] == model][metric].dropna()
        means.append(model_data.mean())
        stds.append(model_data.std())

    x = np.arange(len(model_order))
    colors = sns.color_palette("husl", len(model_order))

    bars = ax.bar(
        x, means, yerr=stds, capsize=5, alpha=0.7, color=colors, edgecolor="black"
    )

    # Highlight best performer
    best_idx = np.argmax(means) if "lpips" not in metric else np.argmin(means)
    bars[best_idx].set_edgecolor("red")
    bars[best_idx].set_linewidth(2.5)

    ax.set_ylabel(ylabel)
    ax.set_xlabel("Model Architecture")
    ax.set_xticks(x)
    ax.set_xticklabels([model_labels[m] for m in model_order], rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for i, (mean, std) in enumerate(zip(means, stds)):
        ax.text(
            i, mean + std + 0.01, f"{mean:.3f}", ha="center", va="bottom", fontsize=8
        )

plt.tight_layout()
plt.savefig(output_dir / "metrics_barplot.png", bbox_inches="tight")
print(f"  Saved: metrics_barplot.png")
plt.close()

# ============================================================================
# 3. VIOLIN PLOTS - Distribution with Density
# ============================================================================
print("\n3. Creating violin plots...")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(
    "Metric Distributions by Model Architecture", fontsize=14, fontweight="bold"
)

for idx, (metric, ylabel) in enumerate(bar_metrics):
    ax = axes[idx]

    # Prepare data for seaborn
    plot_df = df[["model_name", metric]].copy()
    plot_df["model_name"] = plot_df["model_name"].map(model_labels)

    sns.violinplot(
        data=plot_df,
        x="model_name",
        y=metric,
        ax=ax,
        order=[model_labels[m] for m in model_order],
        palette="husl",
        inner="box",
    )

    ax.set_ylabel(ylabel)
    ax.set_xlabel("Model Architecture")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / "metrics_violinplot.png", bbox_inches="tight")
print(f"  Saved: metrics_violinplot.png")
plt.close()

# ============================================================================
# 4. CORRELATION HEATMAP
# ============================================================================
print("\n4. Creating correlation heatmap...")

# Select numeric columns for correlation
metric_cols = [
    "final_ssim",
    "final_psnr",
    "final_mse",
    "final_lpips",
    "training_time_seconds",
    "loss_final",
]
corr_df = df[metric_cols].corr()

# Rename for better labels
label_map = {
    "final_ssim": "SSIM",
    "final_psnr": "PSNR",
    "final_mse": "MSE",
    "final_lpips": "LPIPS",
    "training_time_seconds": "Training Time",
    "loss_final": "Final Loss",
}
corr_df = corr_df.rename(columns=label_map, index=label_map)

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(
    corr_df,
    annot=True,
    fmt=".3f",
    cmap="coolwarm",
    center=0,
    square=True,
    linewidths=1,
    cbar_kws={"shrink": 0.8},
    ax=ax,
)
ax.set_title(
    "Correlation Matrix of Performance Metrics", fontsize=14, fontweight="bold", pad=20
)
plt.tight_layout()
plt.savefig(output_dir / "correlation_heatmap.png", bbox_inches="tight")
print(f"  Saved: correlation_heatmap.png")
plt.close()

# ============================================================================
# 5. LOSS CONVERGENCE CURVES (Sample)
# ============================================================================
print("\n5. Creating loss convergence curves...")

# Load a few sample experiments to show convergence
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Training Loss Convergence Examples", fontsize=14, fontweight="bold")

# Select one experiment per model (pair_id=1 for consistency)
for model_idx, model in enumerate(["vgg19", "resnet50", "inception_v3"]):
    # Find metadata file
    metadata_files = list(
        Path("outputs/full_batch_experiments").glob(f"{model}/pair_1/metadata.json")
    )

    if metadata_files:
        with open(metadata_files[0], "r") as f:
            metadata = json.load(f)

        loss_tracking = metadata.get("loss_tracking", {})

        # Total loss - handle both old and new formats
        if "total_losses" in loss_tracking:
            total_losses_data = loss_tracking["total_losses"]
            # Check if it's list of dicts (new format) or list of values (old format)
            if total_losses_data and isinstance(total_losses_data[0], dict):
                epochs = [item["epoch"] for item in total_losses_data]
                total_losses = [item["value"] for item in total_losses_data]
            else:
                total_losses = total_losses_data
                epochs = list(range(1, len(total_losses) + 1))

            axes[0].plot(
                epochs,
                total_losses,
                label=model_labels[model],
                alpha=0.7,
                linewidth=1.5,
            )

        # Content vs Style loss
        if (
            "content_losses" in loss_tracking
            and "style_losses" in loss_tracking
            and model == "resnet50"
        ):
            content_losses_data = loss_tracking["content_losses"]
            style_losses_data = loss_tracking["style_losses"]

            # Handle new format
            if content_losses_data and isinstance(content_losses_data[0], dict):
                epochs = [item["epoch"] for item in content_losses_data]
                content_losses = [item["value"] for item in content_losses_data]
                style_losses = [item["value"] for item in style_losses_data]
            else:
                content_losses = content_losses_data
                style_losses = style_losses_data
                epochs = list(range(1, len(content_losses) + 1))

            axes[1].plot(
                epochs, content_losses, label="Content Loss", alpha=0.7, linewidth=1.5
            )
            axes[1].plot(
                epochs, style_losses, label="Style Loss", alpha=0.7, linewidth=1.5
            )

axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Total Loss")
axes[0].set_title("Total Loss Convergence")
axes[0].legend()
axes[0].grid(alpha=0.3)
axes[0].set_yscale("log")

axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss Value")
axes[1].set_title("Content vs Style Loss (ResNet50 - Pair 1)")
axes[1].legend()
axes[1].grid(alpha=0.3)
axes[1].set_yscale("log")

plt.tight_layout()
plt.savefig(output_dir / "loss_convergence.png", bbox_inches="tight")
print(f"  Saved: loss_convergence.png")
plt.close()

# ============================================================================
# 6. PERFORMANCE SUMMARY RADAR CHART
# ============================================================================
print("\n6. Creating radar chart...")

from math import pi

# Normalize metrics to 0-1 scale for radar chart
metrics_for_radar = ["final_ssim", "final_psnr", "final_lpips"]
categories = ["SSIM", "PSNR", "LPIPS\n(inverted)"]

fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection="polar"))

# Calculate mean for each model
angles = [n / len(categories) * 2 * pi for n in range(len(categories))]
angles += angles[:1]

for model in model_order:
    model_data = df[df["model_name"] == model]

    values = []
    # SSIM (0-1, higher better) - use as is
    values.append(model_data["final_ssim"].mean())
    # PSNR (normalize to 0-1, higher better)
    psnr_norm = (model_data["final_psnr"].mean() - 7) / (15 - 7)  # Approx range 7-15
    values.append(max(0, min(1, psnr_norm)))
    # LPIPS (0-1, lower better) - invert
    values.append(1 - model_data["final_lpips"].mean())

    values += values[:1]

    ax.plot(angles, values, "o-", linewidth=2, label=model_labels[model])
    ax.fill(angles, values, alpha=0.15)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, size=11)
ax.set_ylim(0, 1)
ax.set_title(
    "Multi-Metric Performance Comparison\n(Normalized to 0-1 scale)",
    size=14,
    fontweight="bold",
    pad=20,
)
ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
ax.grid(True)

plt.tight_layout()
plt.savefig(output_dir / "performance_radar.png", bbox_inches="tight")
print(f"  Saved: performance_radar.png")
plt.close()

print("\n" + "=" * 80)
print("ALL VISUALIZATIONS GENERATED!")
print("=" * 80)
print(f"\nOutput directory: {output_dir.absolute()}")
print("\nGenerated files:")
print("  1. metrics_boxplots.png - Box plots for all metrics")
print("  2. metrics_barplot.png - Bar charts with error bars")
print("  3. metrics_violinplot.png - Violin plots showing distributions")
print("  4. correlation_heatmap.png - Correlation matrix")
print("  5. loss_convergence.png - Training loss curves")
print("  6. performance_radar.png - Multi-metric radar chart")
print("=" * 80)
