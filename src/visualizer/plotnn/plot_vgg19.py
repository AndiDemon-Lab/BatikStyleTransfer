"""
VGG19 Architecture Visualization for Neural Style Transfer
Using PlotNeuralNet to generate high-quality LaTeX diagrams
"""

import sys
sys.path.append('PlotNeuralNet')

from PlotNeuralNet.PyCore.TikzGen import (
    ToBegin,
    ToConnection,
    ToConv,
    ToConvConvRelu,
    ToCor,
    ToEnd,
    ToGenerate,
    ToHead,
    ToPool,
)

# VGG19 Architecture for NST
# 5 blocks with increasing channels: 64, 128, 256, 512, 512
arch = [
    ToHead("PlotNeuralNet/PlotNeuralNet"),
    ToCor(),
    ToBegin(),
    
    # Input
    ToConv("input", sFilter=224, nFilter=3, offset="(0,0,0)", to="(0,0,0)", 
           width=1, height=40, depth=40, caption="Input"),
    
    # Block 1: 64 channels, 2 conv layers
    ToConvConvRelu("conv1", sFilter=224, nFilter=(64, 64), offset="(1.5,0,0)", 
                   to="(input-east)", width=(2, 2), height=40, depth=40, caption="Conv1"),
    ToConnection("input", "conv1"),
    ToPool("pool1", offset="(0,0,0)", to="(conv1-east)", width=1, height=35, depth=35, opacity=0.5),
    
    # Block 2: 128 channels, 2 conv layers
    ToConvConvRelu("conv2", sFilter=112, nFilter=(128, 128), offset="(1.5,0,0)", 
                   to="(pool1-east)", width=(3, 3), height=35, depth=35, caption="Conv2"),
    ToConnection("pool1", "conv2"),
    ToPool("pool2", offset="(0,0,0)", to="(conv2-east)", width=1, height=30, depth=30, opacity=0.5),
    
    # Block 3: 256 channels, 4 conv layers
    ToConv("conv3_1", sFilter=56, nFilter=256, offset="(1.5,0,0)", to="(pool2-east)", 
           width=4, height=30, depth=30, caption="Conv3\\_1"),
    ToConnection("pool2", "conv3_1"),
    ToConv("conv3_2", sFilter=56, nFilter=256, offset="(0,0,0)", to="(conv3_1-east)", 
           width=4, height=30, depth=30, caption="Conv3\\_2"),
    ToConv("conv3_3", sFilter=56, nFilter=256, offset="(0,0,0)", to="(conv3_2-east)", 
           width=4, height=30, depth=30, caption="Conv3\\_3"),
    ToConv("conv3_4", sFilter=56, nFilter=256, offset="(0,0,0)", to="(conv3_3-east)", 
           width=4, height=30, depth=30, caption="Conv3\\_4"),
    ToPool("pool3", offset="(0,0,0)", to="(conv3_4-east)", width=1, height=25, depth=25, opacity=0.5),
    
    # Block 4: 512 channels, 4 conv layers
    ToConv("conv4_1", sFilter=28, nFilter=512, offset="(1.5,0,0)", to="(pool3-east)", 
           width=6, height=25, depth=25, caption="Conv4\\_1"),
    ToConnection("pool3", "conv4_1"),
    ToConv("conv4_2", sFilter=28, nFilter=512, offset="(0,0,0)", to="(conv4_1-east)", 
           width=6, height=25, depth=25, caption="Conv4\\_2"),
    ToConv("conv4_3", sFilter=28, nFilter=512, offset="(0,0,0)", to="(conv4_2-east)", 
           width=6, height=25, depth=25, caption="Conv4\\_3"),
    ToConv("conv4_4", sFilter=28, nFilter=512, offset="(0,0,0)", to="(conv4_3-east)", 
           width=6, height=25, depth=25, caption="Conv4\\_4"),
    ToPool("pool4", offset="(0,0,0)", to="(conv4_4-east)", width=1, height=20, depth=20, opacity=0.5),
    
    # Block 5: 512 channels, 4 conv layers
    ToConv("conv5_1", sFilter=14, nFilter=512, offset="(1.5,0,0)", to="(pool4-east)", 
           width=6, height=20, depth=20, caption="Conv5\\_1"),
    ToConnection("pool4", "conv5_1"),
    ToConv("conv5_2", sFilter=14, nFilter=512, offset="(0,0,0)", to="(conv5_1-east)", 
           width=6, height=20, depth=20, caption="Conv5\\_2"),
    ToConv("conv5_3", sFilter=14, nFilter=512, offset="(0,0,0)", to="(conv5_2-east)", 
           width=6, height=20, depth=20, caption="Conv5\\_3"),
    ToConv("conv5_4", sFilter=14, nFilter=512, offset="(0,0,0)", to="(conv5_3-east)", 
           width=6, height=20, depth=20, caption="Conv5\\_4"),
    ToPool("pool5", offset="(0,0,0)", to="(conv5_4-east)", width=1, height=15, depth=15, opacity=0.5),
    
    # Output features for NST
    ToConv("features", sFilter=7, nFilter=512, offset="(1.5,0,0)", to="(pool5-east)", 
           width=8, height=15, depth=15, caption="Features"),
    ToConnection("pool5", "features"),
    
    ToEnd(),
]


def main():
    """Generate VGG19 architecture diagram"""
    ToGenerate(arch, "outputs/model_visualizations/vgg19_arch.tex")
    print("[OK] Generated VGG19 LaTeX file: outputs/model_visualizations/vgg19_arch.tex")


if __name__ == "__main__":
    main()
