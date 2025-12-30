"""
ResNet101 Architecture Visualization for Neural Style Transfer
Using PlotNeuralNet to generate high-quality LaTeX diagrams
"""

import sys
sys.path.append('PlotNeuralNet')

from PlotNeuralNet.PyCore.TikzGen import (
    ToBegin,
    ToConnection,
    ToConv,
    ToConvRes,
    ToCor,
    ToEnd,
    ToGenerate,
    ToHead,
    ToPool,
)

# ResNet101 Architecture for NST
# Initial conv + 4 residual blocks with [3, 4, 23, 3] bottleneck layers
arch = [
    ToHead("PlotNeuralNet/PlotNeuralNet"),
    ToCor(),
    ToBegin(),
    
    # Input
    ToConv("input", sFilter=224, nFilter=3, offset="(0,0,0)", to="(0,0,0)", 
           width=1, height=40, depth=40, caption="Input"),
    
    # Initial Conv + BN + ReLU + MaxPool
    ToConv("conv1", sFilter=112, nFilter=64, offset="(2,0,0)", to="(input-east)", 
           width=2, height=38, depth=38, caption="Conv1 7x7"),
    ToConnection("input", "conv1"),
    ToPool("pool1", offset="(0,0,0)", to="(conv1-east)", width=1, height=35, depth=35, opacity=0.5),
    
    # Block 1: conv2_x (3 bottleneck layers, 64->256 channels)
    ToConvRes("res2", sFilter=56, nFilter=256, offset="(2,0,0)", to="(pool1-east)", 
              width=5, height=35, depth=35, opacity=0.3, caption="Res2 (3x)"),
    ToConnection("pool1", "res2"),
    
    # Block 2: conv3_x (4 bottleneck layers, 256->512 channels)
    ToConvRes("res3", sFilter=28, nFilter=512, offset="(2,0,0)", to="(res2-east)", 
              width=6, height=30, depth=30, opacity=0.3, caption="Res3 (4x)"),
    ToConnection("res2", "res3"),
    
    # Block 3: conv4_x (23 bottleneck layers, 512->1024 channels) - MAIN DIFFERENCE
    ToConvRes("res4", sFilter=14, nFilter=1024, offset="(2,0,0)", to="(res3-east)", 
              width=10, height=25, depth=25, opacity=0.3, caption="Res4 (23x)"),
    ToConnection("res3", "res4"),
    
    # Block 4: conv5_x (3 bottleneck layers, 1024->2048 channels)
    ToConvRes("res5", sFilter=7, nFilter=2048, offset="(2,0,0)", to="(res4-east)", 
              width=9, height=20, depth=20, opacity=0.3, caption="Res5 (3x)"),
    ToConnection("res4", "res5"),
    
    # Output features for NST
    ToConv("features", sFilter=7, nFilter=2048, offset="(2,0,0)", to="(res5-east)", 
           width=10, height=18, depth=18, caption="Features"),
    ToConnection("res5", "features"),
    
    ToEnd(),
]


def main():
    """Generate ResNet101 architecture diagram"""
    ToGenerate(arch, "outputs/model_visualizations/resnet101_arch.tex")
    print("[OK] Generated ResNet101 LaTeX file: outputs/model_visualizations/resnet101_arch.tex")


if __name__ == "__main__":
    main()
