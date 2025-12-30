"""
Generate visual comparison grid for NST outputs.
Shows content, style, and outputs from all 5 models for pair_1.
"""

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import numpy as np

# Set publication-quality style
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300


def resize_to_square(img, size=512):
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

print("Creating visual comparison for pair_1...")

# Load images
pair_dir = Path("outputs/hasil/1")
content_img = resize_to_square(Image.open(pair_dir / "content.jpg"))
style_img = resize_to_square(Image.open(pair_dir / "style.jpg"))

# Load outputs from each model
outputs = {}
for model in models:
    output_path = Path(
        f"outputs/full_batch_experiments/{model}/pair_1/final_output.png"
    )
    if output_path.exists():
        outputs[model] = resize_to_square(Image.open(output_path))
        print(f"  Loaded {model}")
    else:
        print(f"Missing {model}")

# Create figure with grid layout
# Layout: 2 rows x 4 columns
# Row 1: Content, Style, VGG16, VGG19
# Row 2: Inception V3, ResNet50, ResNet101, (empty)
fig = plt.figure(figsize=(16, 8))
gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.3, wspace=0.2)

# Row 1
ax1 = fig.add_subplot(gs[0, 0])
ax1.imshow(content_img)
ax1.set_title("Content Image", fontsize=12, fontweight="bold")
ax1.axis("off")

ax2 = fig.add_subplot(gs[0, 1])
ax2.imshow(style_img)
ax2.set_title("Style Image", fontsize=12, fontweight="bold")
ax2.axis("off")

ax3 = fig.add_subplot(gs[0, 2])
if "vgg16" in outputs:
    ax3.imshow(outputs["vgg16"])
ax3.set_title("VGG16", fontsize=12, fontweight="bold", color="#1f77b4")
ax3.axis("off")

ax4 = fig.add_subplot(gs[0, 3])
if "vgg19" in outputs:
    ax4.imshow(outputs["vgg19"])
ax4.set_title("VGG19", fontsize=12, fontweight="bold", color="#ff7f0e")
ax4.axis("off")

# Row 2
ax5 = fig.add_subplot(gs[1, 0])
if "inception_v3" in outputs:
    ax5.imshow(outputs["inception_v3"])
ax5.set_title("Inception V3", fontsize=12, fontweight="bold", color="#2ca02c")
ax5.axis("off")

ax6 = fig.add_subplot(gs[1, 1])
if "resnet50" in outputs:
    ax6.imshow(outputs["resnet50"])
ax6.set_title("ResNet50", fontsize=12, fontweight="bold", color="#d62728")
ax6.axis("off")

ax7 = fig.add_subplot(gs[1, 2])
if "resnet101" in outputs:
    ax7.imshow(outputs["resnet101"])
ax7.set_title("ResNet101", fontsize=12, fontweight="bold", color="#9467bd")
ax7.axis("off")

# Add overall title
fig.suptitle(
    "Neural Style Transfer Results Comparison - Pair 1",
    fontsize=16,
    fontweight="bold",
    y=0.98,
)

plt.savefig(output_dir / "visual_comparison_pair1.png", bbox_inches="tight", dpi=300)
print(f"\nSaved: visual_comparison_pair1.png")
plt.close()

print("\n" + "=" * 80)
print("VISUAL COMPARISON GENERATED!")
print("=" * 80)
print(f"\nOutput: {output_dir.absolute()}/visual_comparison_pair1.png")
print("Layout: 2x4 grid with content, style, and 5 model outputs")
print("All images resized to 512x512 square format")
print("=" * 80)
