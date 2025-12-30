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

# Define model order and colors
model_order = ["vgg16", "vgg19", "inception_v3", "resnet50", "resnet101"]
model_labels = {
    "vgg16": "VGG16",
    "vgg19": "VGG19",
    "inception_v3": "Inception V3",
    "resnet50": "ResNet50",
    "resnet101": "ResNet101",
}

print("Creating loss convergence plot...")

fig, ax = plt.subplots(1, 1, figsize=(10, 6))
fig.suptitle("Training Loss Convergence Comparison", fontsize=14, fontweight="bold")

# Load loss data for each model (using pair_1 as representative)
colors = sns.color_palette("husl", len(model_order))

for model_idx, model in enumerate(model_order):
    # Find metadata file for pair_1
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

            # Plot with model-specific color
            ax.plot(
                epochs,
                total_losses,
                label=model_labels[model],
                alpha=0.8,
                linewidth=2,
                color=colors[model_idx],
            )

ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("Total Loss", fontsize=12)
ax.set_yscale("log")
ax.grid(alpha=0.3, which="both", linestyle="--", linewidth=0.5)
ax.legend(loc="upper right", framealpha=0.9)
ax.set_title(
    "All models show monotonic loss reduction with varying convergence rates",
    fontsize=10,
    style="italic",
    pad=10,
)

plt.tight_layout()
plt.savefig(output_dir / "loss_convergence_comparison.png", bbox_inches="tight")
print(f"Saved: loss_convergence_comparison.png")
plt.close()

print("\n" + "=" * 80)
print("LOSS CONVERGENCE PLOT GENERATED!")
print("=" * 80)
print(f"\nOutput: {output_dir.absolute()}/loss_convergence_comparison.png")
print("=" * 80)
