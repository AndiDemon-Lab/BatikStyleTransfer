"""
Complete PlotNeuralNet Visualization Generator
Generates all NST model architectures using the correct PlotNeuralNet API
"""

import sys
import os

# Add PlotNeuralNet to path
sys.path.insert(0, 'PlotNeuralNet/PlotNeuralNet')

from PyCore.TikzGen import (
    ToHead,
    ToCor,
    ToBegin,
    ToConv,
    ToPool,
    ToConnection,
    ToConvRes,
    ToEnd,
    ToGenerate
)

OUTPUT_DIR = "outputs/model_visualizations"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_vgg16():
    """Generate VGG16 architecture"""
    arch = [
        ToHead('PlotNeuralNet/PlotNeuralNet'),
        ToCor(),
        ToBegin(),
        
        # Block 1 - 64 channels
        ToConv("conv1_1", sFilter=224, nFilter=64, offset="(0,0,0)", to="(0,0,0)", height=40, depth=40, width=2, caption="Conv1\\_1"),
        ToConv("conv1_2", sFilter=224, nFilter=64, offset="(1,0,0)", to="(conv1_1-east)", height=40, depth=40, width=2, caption="Conv1\\_2"),
        ToPool("pool1", offset="(0,0,0)", to="(conv1_2-east)", height=35, depth=35, width=1),
        ToConnection("conv1_2", "pool1"),
        
        # Block 2 - 128 channels
        ToConv("conv2_1", sFilter=112, nFilter=128, offset="(1,0,0)", to="(pool1-east)", height=35, depth=35, width=3, caption="Conv2\\_1"),
        ToConnection("pool1", "conv2_1"),
        ToConv("conv2_2", sFilter=112, nFilter=128, offset="(1,0,0)", to="(conv2_1-east)", height=35, depth=35, width=3, caption="Conv2\\_2"),
        ToPool("pool2", offset="(0,0,0)", to="(conv2_2-east)", height=30, depth=30, width=1),
        ToConnection("conv2_2", "pool2"),
        
        # Block 3 - 256 channels (3 layers)
        ToConv("conv3_1", sFilter=56, nFilter=256, offset="(1,0,0)", to="(pool2-east)", height=30, depth=30, width=4, caption="Conv3\\_1"),
        ToConnection("pool2", "conv3_1"),
        ToConv("conv3_2", sFilter=56, nFilter=256, offset="(1,0,0)", to="(conv3_1-east)", height=30, depth=30, width=4, caption="Conv3\\_2"),
        ToConv("conv3_3", sFilter=56, nFilter=256, offset="(1,0,0)", to="(conv3_2-east)", height=30, depth=30, width=4, caption="Conv3\\_3"),
        ToPool("pool3", offset="(0,0,0)", to="(conv3_3-east)", height=25, depth=25, width=1),
        ToConnection("conv3_3", "pool3"),
        
        # Block 4 - 512 channels (3 layers)
        ToConv("conv4_1", sFilter=28, nFilter=512, offset="(1,0,0)", to="(pool3-east)", height=25, depth=25, width=6, caption="Conv4\\_1"),
        ToConnection("pool3", "conv4_1"),
        ToConv("conv4_2", sFilter=28, nFilter=512, offset="(1,0,0)", to="(conv4_1-east)", height=25, depth=25, width=6, caption="Conv4\\_2"),
        ToConv("conv4_3", sFilter=28, nFilter=512, offset="(1,0,0)", to="(conv4_2-east)", height=25, depth=25, width=6, caption="Conv4\\_3"),
        ToPool("pool4", offset="(0,0,0)", to="(conv4_3-east)", height=20, depth=20, width=1),
        ToConnection("conv4_3", "pool4"),
        
        # Block 5 - 512 channels (3 layers)
        ToConv("conv5_1", sFilter=14, nFilter=512, offset="(1,0,0)", to="(pool4-east)", height=20, depth=20, width=6, caption="Conv5\\_1"),
        ToConnection("pool4", "conv5_1"),
        ToConv("conv5_2", sFilter=14, nFilter=512, offset="(1,0,0)", to="(conv5_1-east)", height=20, depth=20, width=6, caption="Conv5\\_2"),
        ToConv("conv5_3", sFilter=14, nFilter=512, offset="(1,0,0)", to="(conv5_2-east)", height=20, depth=20, width=6, caption="Conv5\\_3"),
        
        ToEnd()
    ]
    
    ToGenerate(arch, f"{OUTPUT_DIR}/vgg16_final.tex")
    print("[OK] Generated VGG16")


def generate_vgg19():
    """Generate VGG19 architecture"""
    arch = [
        ToHead('PlotNeuralNet/PlotNeuralNet'),
        ToCor(),
        ToBegin(),
        
        # Block 1 - 64 channels
        ToConv("conv1_1", sFilter=224, nFilter=64, offset="(0,0,0)", to="(0,0,0)", height=40, depth=40, width=2, caption="Conv1\\_1"),
        ToConv("conv1_2", sFilter=224, nFilter=64, offset="(1,0,0)", to="(conv1_1-east)", height=40, depth=40, width=2, caption="Conv1\\_2"),
        ToPool("pool1", offset="(0,0,0)", to="(conv1_2-east)", height=35, depth=35, width=1),
        ToConnection("conv1_2", "pool1"),
        
        # Block 2 - 128 channels
        ToConv("conv2_1", sFilter=112, nFilter=128, offset="(1,0,0)", to="(pool1-east)", height=35, depth=35, width=3, caption="Conv2\\_1"),
        ToConnection("pool1", "conv2_1"),
        ToConv("conv2_2", sFilter=112, nFilter=128, offset="(1,0,0)", to="(conv2_1-east)", height=35, depth=35, width=3, caption="Conv2\\_2"),
        ToPool("pool2", offset="(0,0,0)", to="(conv2_2-east)", height=30, depth=30, width=1),
        ToConnection("conv2_2", "pool2"),
        
        # Block 3 - 256 channels (4 layers)
        ToConv("conv3_1", sFilter=56, nFilter=256, offset="(1,0,0)", to="(pool2-east)", height=30, depth=30, width=4, caption="Conv3\\_1"),
        ToConnection("pool2", "conv3_1"),
        ToConv("conv3_2", sFilter=56, nFilter=256, offset="(1,0,0)", to="(conv3_1-east)", height=30, depth=30, width=4, caption="Conv3\\_2"),
        ToConv("conv3_3", sFilter=56, nFilter=256, offset="(1,0,0)", to="(conv3_2-east)", height=30, depth=30, width=4, caption="Conv3\\_3"),
        ToConv("conv3_4", sFilter=56, nFilter=256, offset="(1,0,0)", to="(conv3_3-east)", height=30, depth=30, width=4, caption="Conv3\\_4"),
        ToPool("pool3", offset="(0,0,0)", to="(conv3_4-east)", height=25, depth=25, width=1),
        ToConnection("conv3_4", "pool3"),
        
        # Block 4 - 512 channels (4 layers)
        ToConv("conv4_1", sFilter=28, nFilter=512, offset="(1,0,0)", to="(pool3-east)", height=25, depth=25, width=6, caption="Conv4\\_1"),
        ToConnection("pool3", "conv4_1"),
        ToConv("conv4_2", sFilter=28, nFilter=512, offset="(1,0,0)", to="(conv4_1-east)", height=25, depth=25, width=6, caption="Conv4\\_2"),
        ToConv("conv4_3", sFilter=28, nFilter=512, offset="(1,0,0)", to="(conv4_2-east)", height=25, depth=25, width=6, caption="Conv4\\_3"),
        ToConv("conv4_4", sFilter=28, nFilter=512, offset="(1,0,0)", to="(conv4_3-east)", height=25, depth=25, width=6, caption="Conv4\\_4"),
        ToPool("pool4", offset="(0,0,0)", to="(conv4_4-east)", height=20, depth=20, width=1),
        ToConnection("conv4_4", "pool4"),
        
        # Block 5 - 512 channels (4 layers)
        ToConv("conv5_1", sFilter=14, nFilter=512, offset="(1,0,0)", to="(pool4-east)", height=20, depth=20, width=6, caption="Conv5\\_1"),
        ToConnection("pool4", "conv5_1"),
        ToConv("conv5_2", sFilter=14, nFilter=512, offset="(1,0,0)", to="(conv5_1-east)", height=20, depth=20, width=6, caption="Conv5\\_2"),
        ToConv("conv5_3", sFilter=14, nFilter=512, offset="(1,0,0)", to="(conv5_2-east)", height=20, depth=20, width=6, caption="Conv5\\_3"),
        ToConv("conv5_4", sFilter=14, nFilter=512, offset="(1,0,0)", to="(conv5_3-east)", height=20, depth=20, width=6, caption="Conv5\\_4"),
        
        ToEnd()
    ]
    
    ToGenerate(arch, f"{OUTPUT_DIR}/vgg19_final.tex")
    print("[OK] Generated VGG19")


def generate_resnet50():
    """Generate ResNet50 architecture (simplified)"""
    arch = [
        ToHead('PlotNeuralNet/PlotNeuralNet'),
        ToCor(),
        ToBegin(),
        
        # Input
        ToConv("input", sFilter=224, nFilter=3, offset="(0,0,0)", to="(0,0,0)", height=40, depth=40, width=1, caption="Input"),
        
        # Initial conv
        ToConv("conv1", sFilter=112, nFilter=64, offset="(2,0,0)", to="(input-east)", height=38, depth=38, width=2, caption="Conv1"),
        ToConnection("input", "conv1"),
        ToPool("pool1", offset="(0,0,0)", to="(conv1-east)", height=35, depth=35, width=1),
        
        # Residual blocks (simplified representation)
        ToConvRes("res2", sFilter=56, nFilter=256, offset="(2,0,0)", to="(pool1-east)", width=5, height=35, depth=35, caption="Res2 (3x)"),
        ToConnection("pool1", "res2"),
        
        ToConvRes("res3", sFilter=28, nFilter=512, offset="(2,0,0)", to="(res2-east)", width=6, height=30, depth=30, caption="Res3 (4x)"),
        ToConnection("res2", "res3"),
        
        ToConvRes("res4", sFilter=14, nFilter=1024, offset="(2,0,0)", to="(res3-east)", width=8, height=25, depth=25, caption="Res4 (6x)"),
        ToConnection("res3", "res4"),
        
        ToConvRes("res5", sFilter=7, nFilter=2048, offset="(2,0,0)", to="(res4-east)", width=10, height=20, depth=20, caption="Res5 (3x)"),
        ToConnection("res4", "res5"),
        
        ToEnd()
    ]
    
    ToGenerate(arch, f"{OUTPUT_DIR}/resnet50_final.tex")
    print("[OK] Generated ResNet50")


def main():
    print("=" * 70)
    print("PlotNeuralNet Architecture Generator")
    print("=" * 70)
    
    print("\nGenerating LaTeX files...")
    generate_vgg16()
    generate_vgg19()
    generate_resnet50()
    
    print("\n" + "=" * 70)
    print("LaTeX files generated successfully!")
    print(f"Location: {OUTPUT_DIR}/")
    print("=" * 70)
    print("\nNext: Upload these .tex files to Overleaf.com to compile to PDF")


if __name__ == "__main__":
    main()
