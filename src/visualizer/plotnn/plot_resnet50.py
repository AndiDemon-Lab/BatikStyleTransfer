"""
ResNet50 Architecture Visualization for Neural Style Transfer
Using PlotNeuralNet to generate high-quality LaTeX diagrams
"""

import sys
sys.path.append('PlotNeuralNet')

from PlotNeuralNet.PyCore.Blocks import BlockRes
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
    ToSkip,
)

# ResNet50 Architecture for NST
# Initial conv + 4 residual blocks with [3, 4, 6, 3] bottleneck layers
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
    ToConvRes("res2a", sFilter=56, nFilter=256, offset="(2,0,0)", to="(pool1-east)", 
              width=4, height=35, depth=35, opacity=0.3, caption="Res2a"),
    ToConnection("pool1", "res2a"),
    ToConvRes("res2b", sFilter=56, nFilter=256, offset="(0,0,0)", to="(res2a-east)", 
              width=4, height=35, depth=35, opacity=0.3, caption="Res2b"),
    ToConvRes("res2c", sFilter=56, nFilter=256, offset="(0,0,0)", to="(res2b-east)", 
              width=4, height=35, depth=35, opacity=0.3, caption="Res2c"),
    
    # Block 2: conv3_x (4 bottleneck layers, 256->512 channels)
    ToConvRes("res3a", sFilter=28, nFilter=512, offset="(2,0,0)", to="(res2c-east)", 
              width=5, height=30, depth=30, opacity=0.3, caption="Res3a"),
    ToConnection("res2c", "res3a"),
    ToConvRes("res3b", sFilter=28, nFilter=512, offset="(0,0,0)", to="(res3a-east)", 
              width=5, height=30, depth=30, opacity=0.3, caption="Res3b"),
    ToConvRes("res3c", sFilter=28, nFilter=512, offset="(0,0,0)", to="(res3b-east)", 
              width=5, height=30, depth=30, opacity=0.3, caption="Res3c"),
    ToConvRes("res3d", sFilter=28, nFilter=512, offset="(0,0,0)", to="(res3c-east)", 
              width=5, height=30, depth=30, opacity=0.3, caption="Res3d"),
    
    # Block 3: conv4_x (6 bottleneck layers, 512->1024 channels)
    ToConvRes("res4a", sFilter=14, nFilter=1024, offset="(2,0,0)", to="(res3d-east)", 
              width=7, height=25, depth=25, opacity=0.3, caption="Res4a"),
    ToConnection("res3d", "res4a"),
    ToConvRes("res4b", sFilter=14, nFilter=1024, offset="(0,0,0)", to="(res4a-east)", 
              width=7, height=25, depth=25, opacity=0.3, caption="Res4b"),
    ToConvRes("res4c", sFilter=14, nFilter=1024, offset="(0,0,0)", to="(res4b-east)", 
              width=7, height=25, depth=25, opacity=0.3, caption="Res4c"),
    ToConvRes("res4d", sFilter=14, nFilter=1024, offset="(0,0,0)", to="(res4c-east)", 
              width=7, height=25, depth=25, opacity=0.3, caption="Res4d"),
    ToConvRes("res4e", sFilter=14, nFilter=1024, offset="(0,0,0)", to="(res4d-east)", 
              width=7, height=25, depth=25, opacity=0.3, caption="Res4e"),
    ToConvRes("res4f", sFilter=14, nFilter=1024, offset="(0,0,0)", to="(res4e-east)", 
              width=7, height=25, depth=25, opacity=0.3, caption="Res4f"),
    
    # Block 4: conv5_x (3 bottleneck layers, 1024->2048 channels)
    ToConvRes("res5a", sFilter=7, nFilter=2048, offset="(2,0,0)", to="(res4f-east)", 
              width=9, height=20, depth=20, opacity=0.3, caption="Res5a"),
    ToConnection("res4f", "res5a"),
    ToConvRes("res5b", sFilter=7, nFilter=2048, offset="(0,0,0)", to="(res5a-east)", 
              width=9, height=20, depth=20, opacity=0.3, caption="Res5b"),
    ToConvRes("res5c", sFilter=7, nFilter=2048, offset="(0,0,0)", to="(res5b-east)", 
              width=9, height=20, depth=20, opacity=0.3, caption="Res5c"),
    
    # Output features for NST
    ToConv("features", sFilter=7, nFilter=2048, offset="(2,0,0)", to="(res5c-east)", 
           width=10, height=18, depth=18, caption="Features"),
    ToConnection("res5c", "features"),
    
    ToEnd(),
]


def main():
    """Generate ResNet50 architecture diagram"""
    ToGenerate(arch, "outputs/model_visualizations/resnet50_arch.tex")
    print("[OK] Generated ResNet50 LaTeX file: outputs/model_visualizations/resnet50_arch.tex")


if __name__ == "__main__":
    main()
