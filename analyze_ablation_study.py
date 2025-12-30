import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

# Set publication-quality style
plt.style.use("seaborn-v0_8-paper")
sns.set_palette("husl")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 10

output_dir = Path("outputs/visualizations")
output_dir.mkdir(exist_ok=True)

print("=" * 80)
print("ANALYZING HYPERPARAMETER ABLATION STUDY")
print("=" * 80)

# Load Set 1: Weight ratios
print("\n### SET 1: Content/Style Weight Ratios ###")
set1_file = Path(
    "outputs/ablation_study/set1_weights/aggregated_results_20251217_110754.json"
)

if set1_file.exists():
    with open(set1_file, "r") as f:
        set1_data = json.load(f)

    # Extract experiments
    experiments = set1_data.get("experiments", [])
    print(f"Found {len(experiments)} experiments")

    # Create DataFrame
    rows = []
    for exp in experiments:
        config = exp.get("config", {})
        metrics = exp.get("metrics", {})

        row = {
            "pair": exp.get("pair_id", "unknown"),
            "content_weight": config.get("content_weight", 1),
            "style_weight": config.get("style_weight", 1e8),
            "ssim": metrics.get("final_ssim", 0),
            "psnr": metrics.get("final_psnr", 0),
            "lpips": metrics.get("final_lpips", 0),
            "training_time": metrics.get("training_time_seconds", 0),
        }
        rows.append(row)

    df_set1 = pd.DataFrame(rows)

    # Create ratio label
    df_set1["ratio"] = df_set1.apply(
        lambda x: f"{int(x['content_weight'])}:{x['style_weight']:.0e}", axis=1
    )

    print("\nWeight Ratio Results:")
    print(df_set1.groupby("ratio")[["ssim", "psnr", "lpips"]].mean())

    # Visualization: SSIM by weight ratio
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # SSIM
    df_set1.groupby("ratio")["ssim"].mean().plot(
        kind="bar", ax=axes[0], color="steelblue"
    )
    axes[0].set_title("SSIM by Content:Style Ratio")
    axes[0].set_ylabel("SSIM")
    axes[0].set_xlabel("Weight Ratio")
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].tick_params(axis="x", rotation=45)

    # PSNR
    df_set1.groupby("ratio")["psnr"].mean().plot(kind="bar", ax=axes[1], color="coral")
    axes[1].set_title("PSNR by Content:Style Ratio")
    axes[1].set_ylabel("PSNR (dB)")
    axes[1].set_xlabel("Weight Ratio")
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].tick_params(axis="x", rotation=45)

    # LPIPS
    df_set1.groupby("ratio")["lpips"].mean().plot(
        kind="bar", ax=axes[2], color="mediumseagreen"
    )
    axes[2].set_title("LPIPS by Content:Style Ratio")
    axes[2].set_ylabel("LPIPS (lower is better)")
    axes[2].set_xlabel("Weight Ratio")
    axes[2].grid(axis="y", alpha=0.3)
    axes[2].tick_params(axis="x", rotation=45)

    plt.suptitle(
        "Hyperparameter Ablation: Content/Style Weight Ratios",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(output_dir / "ablation_weight_ratios.png", bbox_inches="tight", dpi=300)
    print(f"\nSaved: ablation_weight_ratios.png")
    plt.close()

    # Save summary table
    summary_set1 = df_set1.groupby("ratio")[
        ["ssim", "psnr", "lpips", "training_time"]
    ].agg(["mean", "std"])
    summary_set1.to_csv(output_dir / "ablation_set1_summary.csv")
    print(f"Saved: ablation_set1_summary.csv")

else:
    print(" Set 1 aggregated results not found")

print("\n" + "=" * 80)
print("ABLATION ANALYSIS COMPLETE!")
print("=" * 80)
print(f"\nOutput directory: {output_dir.absolute()}")
print("\nGenerated files:")
print("  1. ablation_weight_ratios.png - Bar charts showing metric variations")
print("  2. ablation_set1_summary.csv - Statistical summary table")
print("=" * 80)
