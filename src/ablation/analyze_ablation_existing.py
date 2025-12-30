"""
Analyze existing ablation study results from set1_weights.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
plt.style.use("seaborn-v0_8-paper")
plt.rcParams["figure.dpi"] = 300

output_dir = Path("outputs/visualizations")
output_dir.mkdir(exist_ok=True)

print("=" * 80)
print("ANALYZING ABLATION STUDY: SET 1 (WEIGHT RATIOS)")
print("=" * 80)

# Load all metadata files
ablation_dir = Path("outputs/ablation_study/set1_weights/vgg19")
metadata_files = list(ablation_dir.glob("pair_*/metadata.json"))

print(f"\nFound {len(metadata_files)} experiments")

# Extract data
rows = []
for mf in metadata_files:
    with open(mf, "r") as f:
        data = json.load(f)

    row = {
        "pair_id": data["pair_id"],
        "style_weight": data["hyperparameters"]["style_weight"],
        "content_weight": data["hyperparameters"]["content_weight"],
        "ssim": data["final_metrics"]["ssim"],
        "psnr": data["final_metrics"]["psnr"],
        "lpips": data["final_metrics"]["lpips"],
        "training_time": data["training_time_seconds"],
    }
    rows.append(row)

df = pd.DataFrame(rows)
df["ratio"] = df.apply(
    lambda x: f"{int(x['content_weight'])}:{x['style_weight']:.0e}", axis=1
)

print("\n### SUMMARY STATISTICS ###")
print(f"\nStyle Weight: {df['style_weight'].iloc[0]:.0e}")
print(f"Content Weight: {df['content_weight'].iloc[0]}")
print(f"\nMean SSIM: {df['ssim'].mean():.4f} ± {df['ssim'].std():.4f}")
print(f"Mean PSNR: {df['psnr'].mean():.2f} ± {df['psnr'].std():.2f}")
print(f"Mean LPIPS: {df['lpips'].mean():.4f} ± {df['lpips'].std():.4f}")
print(
    f"Mean Training Time: {df['training_time'].mean():.1f}s ± {df['training_time'].std():.1f}s"
)

# Save summary
summary = df[["ssim", "psnr", "lpips", "training_time"]].describe()
summary.to_csv(output_dir / "ablation_set1_summary.csv")
print(f"\nSaved: ablation_set1_summary.csv")

print("\n" + "=" * 80)
print("ABLATION ANALYSIS COMPLETE!")
print("=" * 80)
print(f"\nThis is variant: style_weight = 1e7 (reduced style)")
print("Need to compare with baseline (1e8) and other variants (1e9, content=10)")
print("=" * 80)
