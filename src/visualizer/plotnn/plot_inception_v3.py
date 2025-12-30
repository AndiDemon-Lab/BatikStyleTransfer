"""
Inception V3 Architecture Visualization for Neural Style Transfer
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

# Inception V3 Architecture for NST (simplified representation)
# Shows main structure: Initial convs + Inception blocks + Reduction blocks
arch = [
    ToHead("PlotNeuralNet/PlotNeuralNet"),
    ToCor(),
    ToBegin(),
    
    # Input
    ToConv("input", sFilter=299, nFilter=3, offset="(0,0,0)", to="(0,0,0)", 
           width=1, height=42, depth=42, caption="Input"),
    
    # Initial convolutions
    ToConv("conv1", sFilter=149, nFilter=32, offset="(1.5,0,0)", to="(input-east)", 
           width=2, height=40, depth=40, caption="Conv 3x3/2"),
    ToConnection("input", "conv1"),
    
    ToConv("conv2", sFilter=147, nFilter=32, offset="(1,0,0)", to="(conv1-east)", 
           width=2, height=38, depth=38, caption="Conv 3x3"),
    ToConnection("conv1", "conv2"),
    
    ToConv("conv3", sFilter=147, nFilter=64, offset="(1,0,0)", to="(conv2-east)", 
           width=2.5, height=38, depth=38, caption="Conv 3x3"),
    ToConnection("conv2", "conv3"),
    
    ToPool("pool1", offset="(0,0,0)", to="(conv3-east)", width=1, height=35, depth=35, opacity=0.5),
    
    ToConv("conv4", sFilter=73, nFilter=80, offset="(1.5,0,0)", to="(pool1-east)", 
           width=3, height=35, depth=35, caption="Conv 1x1"),
    ToConnection("pool1", "conv4"),
    
    ToConv("conv5", sFilter=71, nFilter=192, offset="(1,0,0)", to="(conv4-east)", 
           width=3.5, height=33, depth=33, caption="Conv 3x3"),
    ToConnection("conv4", "conv5"),
    
    ToPool("pool2", offset="(0,0,0)", to="(conv5-east)", width=1, height=30, depth=30, opacity=0.5),
    
    # Inception-A blocks (3x)
    ToConvRes("incA", sFilter=35, nFilter=288, offset="(2,0,0)", to="(pool2-east)", 
              width=6, height=30, depth=30, opacity=0.4, caption="Inception-A (3x)"),
    ToConnection("pool2", "incA"),
    
    # Reduction-A
    ToConvRes("redA", sFilter=17, nFilter=768, offset="(1.5,0,0)", to="(incA-east)", 
              width=7, height=25, depth=25, opacity=0.5, caption="Reduction-A"),
    ToConnection("incA", "redA"),
    
    # Inception-B blocks (5x)
    ToConvRes("incB", sFilter=17, nFilter=768, offset="(1.5,0,0)", to="(redA-east)", 
              width=8, height=25, depth=25, opacity=0.4, caption="Inception-B (5x)"),
    ToConnection("redA", "incB"),
    
    # Reduction-B
    ToConvRes("redB", sFilter=8, nFilter=1280, offset="(1.5,0,0)", to="(incB-east)", 
              width=9, height=20, depth=20, opacity=0.5, caption="Reduction-B"),
    ToConnection("incB", "redB"),
    
    # Inception-C blocks (2x)
    ToConvRes("incC", sFilter=8, nFilter=2048, offset="(1.5,0,0)", to="(redB-east)", 
              width=10, height=20, depth=20, opacity=0.4, caption="Inception-C (2x)"),
    ToConnection("redB", "incC"),
    
    # Output features for NST
    ToConv("features", sFilter=8, nFilter=2048, offset="(2,0,0)", to="(incC-east)", 
           width=11, height=18, depth=18, caption="Features"),
    ToConnection("incC", "features"),
    
    ToEnd(),
]


def main():
    """Generate Inception V3 architecture diagram"""
    ToGenerate(arch, "outputs/model_visualizations/inception_v3_arch.tex")
    print("[OK] Generated Inception V3 LaTeX file: outputs/model_visualizations/inception_v3_arch.tex")


if __name__ == "__main__":
    main()
