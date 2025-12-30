import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

# Set style
plt.style.use("seaborn-v0_8-paper")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["font.size"] = 10

output_dir = Path("outputs/visualizations")
output_dir.mkdir(exist_ok=True)

print("=" * 80)
print("COMPREHENSIVE ABLATION STUDY ANALYSIS")
print("=" * 80)

# Define all ablation experiments
ablation_sets = {
    "Set 1: Weight Ratios": {
        "baseline": ("outputs/full_batch_experiments/vgg19", "1:1e8 (Baseline)"),
        "variant_a": (
            "outputs/ablation_study/set1_weights/vgg19",
            "1:1e7 (Less style)",
        ),
        # Note: variant_b and variant_c should be in set1_weights with different configs
    },
    "Set 2: Layer Selection": {
        "baseline": ("outputs/full_batch_experiments/vgg19", "L2,L8 (Baseline)"),
        "shallow": (
            "outputs/ablation_study/set2_layers_shallow/vgg19",
            "L1,L6 (Shallow)",
        ),
        # Note: deep and multi-layer variants
    },
}

# Collect all results
all_results = []

print("\n### LOADING EXPERIMENT RESULTS ###\n")

# Load baseline (from full_batch)
baseline_dir = Path("outputs/full_batch_experiments/vgg19")
baseline_pairs = [1, 12, 20]  # Same pairs as ablation

for pair_id in baseline_pairs:
    metadata_file = baseline_dir / f"pair_{pair_id}" / "metadata.json"
    if metadata_file.exists():
        with open(metadata_file, "r") as f:
            data = json.load(f)
        all_results.append(
            {
                "set": "Set 1",
                "variant": "Baseline (1:1e8)",
                "pair_id": pair_id,
                "content_weight": 1,
                "style_weight": 1e8,
                "ssim": data["final_metrics"]["ssim"],
                "psnr": data["final_metrics"]["psnr"],
                "lpips": data["final_metrics"]["lpips"],
                "training_time": data["training_time_seconds"],
            }
        )

print(
    f"✓ Loaded baseline: {len([r for r in all_results if r['variant'] == 'Baseline (1:1e8)'])} experiments"
)

# Load Set 1 variants
set1_dir = Path("outputs/ablation_study/set1_weights/vgg19")
if set1_dir.exists():
    for pair_id in baseline_pairs:
        metadata_file = set1_dir / f"pair_{pair_id}" / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, "r") as f:
                data = json.load(f)

            style_weight = data["hyperparameters"]["style_weight"]
            content_weight = data["hyperparameters"]["content_weight"]

            # Determine variant
            if style_weight == 1e7:
                variant_name = "Variant A (1:1e7)"
            elif style_weight == 1e9:
                variant_name = "Variant B (1:1e9)"
            elif content_weight == 10:
                variant_name = "Variant C (10:1e8)"
            else:
                continue

            all_results.append(
                {
                    "set": "Set 1",
                    "variant": variant_name,
                    "pair_id": pair_id,
                    "content_weight": content_weight,
                    "style_weight": style_weight,
                    "ssim": data["final_metrics"]["ssim"],
                    "psnr": data["final_metrics"]["psnr"],
                    "lpips": data["final_metrics"]["lpips"],
                    "training_time": data["training_time_seconds"],
                }
            )

print(
    f"✓ Loaded Set 1 variants: {len([r for r in all_results if r['set'] == 'Set 1' and 'Variant' in r['variant']])} experiments"
)

# Load Set 2 variants (layers)
set2_shallow = Path("outputs/ablation_study/set2_layers_shallow/vgg19")
if set2_shallow.exists():
    for pair_id in baseline_pairs:
        metadata_file = set2_shallow / f"pair_{pair_id}" / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, "r") as f:
                data = json.load(f)
            all_results.append(
                {
                    "set": "Set 2",
                    "variant": "Shallow (L1,L6)",
                    "pair_id": pair_id,
                    "content_weight": 1,
                    "style_weight": 1e8,
                    "ssim": data["final_metrics"]["ssim"],
                    "psnr": data["final_metrics"]["psnr"],
                    "lpips": data["final_metrics"]["lpips"],
                    "training_time": data["training_time_seconds"],
                }
            )

print(
    f"✓ Loaded Set 2 variants: {len([r for r in all_results if r['set'] == 'Set 2'])} experiments"
)

# Create DataFrame
df = pd.DataFrame(all_results)

print(f"\n### SUMMARY STATISTICS ###")
print(f"\nTotal experiments loaded: {len(df)}")
print(f"Unique variants: {df['variant'].nunique()}")
print(f"\nVariants found:")
for variant in df["variant"].unique():
    count = len(df[df["variant"] == variant])
    print(f"  - {variant}: {count} experiments")

# Generate comparison visualizations
print(f"\n### GENERATING VISUALIZATIONS ###\n")

# 1. SSIM comparison across variants
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Set 1: Weight ratios
set1_data = df[df["set"] == "Set 1"]
if len(set1_data) > 0:
    set1_data.boxplot(column="ssim", by="variant", ax=axes[0])
    axes[0].set_title("Set 1: Content/Style Weight Ratios")
    axes[0].set_xlabel("Configuration")
    axes[0].set_ylabel("SSIM")
    axes[0].get_figure().suptitle("")
    plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha="right")

# Set 2: Layer selection
set2_data = df[df["set"] == "Set 2"]
if len(set2_data) > 0:
    # Add baseline for comparison
    baseline_for_set2 = df[df["variant"] == "Baseline (1:1e8)"].copy()
    baseline_for_set2["set"] = "Set 2"
    baseline_for_set2["variant"] = "Baseline (L2,L8)"
    set2_combined = pd.concat([baseline_for_set2, set2_data])

    set2_combined.boxplot(column="ssim", by="variant", ax=axes[1])
    axes[1].set_title("Set 2: Layer Selection")
    axes[1].set_xlabel("Configuration")
    axes[1].set_ylabel("SSIM")
    axes[1].get_figure().suptitle("")
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha="right")

plt.tight_layout()
plt.savefig(output_dir / "ablation_ssim_comparison.png", bbox_inches="tight", dpi=300)
print("Saved: ablation_ssim_comparison.png")
plt.close()

# 2. Summary table
summary = df.groupby("variant")[["ssim", "psnr", "lpips", "training_time"]].agg(
    ["mean", "std"]
)
summary.to_csv(output_dir / "ablation_summary_table.csv")
print("Saved: ablation_summary_table.csv")

print("\n" + "=" * 80)
print("ABLATION ANALYSIS COMPLETE!")
print("=" * 80)
print(f"\nGenerated files:")
print(f"  1. ablation_ssim_comparison.png")
print(f"  2. ablation_summary_table.csv")
print("=" * 80)
