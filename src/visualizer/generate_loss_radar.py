"""
Generate loss convergence and radar chart (completing visualization suite).
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import json
from math import pi

# Set style
plt.style.use("seaborn-v0_8-paper")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300

output_dir = Path("outputs/visualizations")
output_dir.mkdir(exist_ok=True)

df = pd.read_csv("outputs/main_batch_30samples.csv")

model_order = ["vgg16", "vgg19", "inception_v3", "resnet50", "resnet101"]
model_labels = {
    "vgg16": "VGG16",
    "vgg19": "VGG19",
    "inception_v3": "Inception V3",
    "resnet50": "ResNet50",
    "resnet101": "ResNet101",
}

# ============================================================================
# 5. LOSS CONVERGENCE CURVES (Fixed)
# ============================================================================
print("Creating loss convergence curves...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Training Loss Convergence Examples", fontsize=14, fontweight="bold")

# Select one experiment per model (pair_id=1 for consistency)
for model in ["vgg19", "resnet50", "inception_v3"]:
    metadata_files = list(
        Path("outputs/full_batch_experiments").glob(f"{model}/pair_1/metadata.json")
    )

    if metadata_files:
        with open(metadata_files[0], "r") as f:
            metadata = json.load(f)

        loss_tracking = metadata.get("loss_tracking", {})

        # Extract total losses (handle list of dicts format)
        if "total_losses" in loss_tracking:
            total_losses_data = loss_tracking["total_losses"]

            # Check if it's list of dicts or list of values
            if isinstance(total_losses_data, list) and len(total_losses_data) > 0:
                if isinstance(total_losses_data[0], dict):
                    # Extract values from dicts
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

        # Content vs Style loss for ResNet50
        if model == "resnet50":
            content_losses_data = loss_tracking.get("content_losses", [])
            style_losses_data = loss_tracking.get("style_losses", [])

            if content_losses_data and style_losses_data:
                # Extract values
                if isinstance(content_losses_data[0], dict):
                    content_losses = [item["value"] for item in content_losses_data]
                    style_losses = [item["value"] for item in style_losses_data]
                else:
                    content_losses = content_losses_data
                    style_losses = style_losses_data

                epochs = list(range(1, len(content_losses) + 1))
                axes[1].plot(
                    epochs,
                    content_losses,
                    label="Content Loss",
                    alpha=0.7,
                    linewidth=1.5,
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
print(f"Saved: loss_convergence.png")
plt.close()

# ============================================================================
# 6. PERFORMANCE SUMMARY RADAR CHART
# ============================================================================
print("Creating radar chart...")

categories = ["SSIM", "PSNR", "LPIPS\n(inverted)"]
angles = [n / len(categories) * 2 * pi for n in range(len(categories))]
angles += angles[:1]

fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection="polar"))

for model in model_order:
    model_data = df[df["model_name"] == model]

    values = []
    # SSIM (0-1, higher better)
    values.append(model_data["final_ssim"].mean())
    # PSNR (normalize to 0-1)
    psnr_norm = (model_data["final_psnr"].mean() - 7) / (15 - 7)
    values.append(max(0, min(1, psnr_norm)))
    # LPIPS (invert)
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
print(f"Saved: performance_radar.png")
plt.close()

print("\n" + "=" * 80)
print("ALL VISUALIZATIONS COMPLETE!")
print("=" * 80)
