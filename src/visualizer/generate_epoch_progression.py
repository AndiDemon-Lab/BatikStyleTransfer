from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import numpy as np

# Set publication-quality style
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300


def resize_to_square(img, size=400):
    """Resize image to square by center cropping."""
    width, height = img.size

    # Crop to square (center crop)
    if width > height:
        left = (width - height) // 2
        img = img.crop((left, 0, left + height, height))
    elif height > width:
        top = (height - width) // 2
        img = img.crop((0, top, width, top + width))

    # Resize to target size
    img = img.resize((size, size), Image.Resampling.LANCZOS)
    return img


# Output directory
output_dir = Path("outputs/visualizations")
output_dir.mkdir(exist_ok=True)

# Model order
models = ["vgg16", "vgg19", "inception_v3", "resnet50", "resnet101"]
model_labels = {
    "vgg16": "VGG16",
    "vgg19": "VGG19",
    "inception_v3": "Inception V3",
    "resnet50": "ResNet50",
    "resnet101": "ResNet101",
}

print("Creating epoch progression comparison for pair_1...")

# Load content and style
pair_dir = Path("outputs/hasil/1")
content_img = resize_to_square(Image.open(pair_dir / "content.jpg"))
style_img = resize_to_square(Image.open(pair_dir / "style.jpg"))

# Load epoch images for each model
epoch_1500 = {}
epoch_5000 = {}

for model in models:
    base_path = Path(f"outputs/full_batch_experiments/{model}/pair_1")

    # Epoch 1500
    path_1500 = base_path / "epoch_1500.png"
    if path_1500.exists():
        epoch_1500[model] = resize_to_square(Image.open(path_1500))
        print(f"  Loaded {model} epoch 1500")
    else:
        print(f"Missing {model} epoch 1500")

    # Epoch 5000 (final)
    path_5000 = base_path / "final_output.png"
    if path_5000.exists():
        epoch_5000[model] = resize_to_square(Image.open(path_5000))
        print(f"  Loaded {model} epoch 5000")
    else:
        print(f"Missing {model} epoch 5000")

# Create figure with grid layout
# Layout: 7 rows x 2 columns
# Row 0: Content, Style
# Rows 1-5: Each model (epoch 1500, epoch 5000)
fig = plt.figure(figsize=(10, 18))
gs = gridspec.GridSpec(7, 2, figure=fig, hspace=0.15, wspace=0.1)

# Row 0: Content and Style
ax_content = fig.add_subplot(gs[0, 0])
ax_content.imshow(content_img)
ax_content.set_title("Content Image", fontsize=11, fontweight="bold")
ax_content.axis("off")

ax_style = fig.add_subplot(gs[0, 1])
ax_style.imshow(style_img)
ax_style.set_title("Style Image (Batik)", fontsize=11, fontweight="bold")
ax_style.axis("off")

# Rows 1-5: Each model
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

for idx, model in enumerate(models):
    row = idx + 1

    # Epoch 1500
    ax_1500 = fig.add_subplot(gs[row, 0])
    if model in epoch_1500:
        ax_1500.imshow(epoch_1500[model])
    ax_1500.set_title(
        f"{model_labels[model]} - Epoch 1500",
        fontsize=10,
        fontweight="bold",
        color=colors[idx],
    )
    ax_1500.axis("off")

    # Epoch 5000
    ax_5000 = fig.add_subplot(gs[row, 1])
    if model in epoch_5000:
        ax_5000.imshow(epoch_5000[model])
    ax_5000.set_title(
        f"{model_labels[model]} - Epoch 5000 (Final)",
        fontsize=10,
        fontweight="bold",
        color=colors[idx],
    )
    ax_5000.axis("off")

# Add overall title
fig.suptitle(
    "Training Progression: Epoch 1500 vs 5000 - Pair 1",
    fontsize=14,
    fontweight="bold",
    y=0.995,
)

plt.savefig(output_dir / "epoch_progression_pair1.png", bbox_inches="tight", dpi=300)
print(f"\nSaved: epoch_progression_pair1.png")
plt.close()

print("\n" + "=" * 80)
print("EPOCH PROGRESSION COMPARISON GENERATED!")
print("=" * 80)
print(f"\nOutput: {output_dir.absolute()}/epoch_progression_pair1.png")
print("Layout: 7x2 grid showing evolution from epoch 1500 to 5000")
print("All images resized to 400x400 square format")
print("=" * 80)
