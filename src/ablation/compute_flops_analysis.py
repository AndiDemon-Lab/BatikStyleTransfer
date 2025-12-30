"""
Computational Cost Analysis for Neural Style Transfer
Measures FLOPs, VRAM usage, parameters, and timing for all architectures
"""

import torch
import torch.nn as nn
import time
import json
import pandas as pd
from pathlib import Path
import gc

# Try to import thop for FLOPs calculation
try:
    from thop import profile, clever_format

    HAS_THOP = True
except ImportError:
    HAS_THOP = False
    print("Warning: thop not installed. Install with: pip install thop")
    print("FLOPs calculation will be skipped.")

from src.models import NeuralStyleTransfer
from PIL import Image
from torchvision import transforms


class ComputationalAnalyzer:
    """Analyze computational requirements for NST models"""

    def __init__(self, image_size=512, device="cuda"):
        self.image_size = image_size
        self.device = device if torch.cuda.is_available() else "cpu"
        self.results = []

        # Standard image preprocessing
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def count_parameters(self, model):
        """Count total and trainable parameters"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total_params, trainable_params

    def measure_flops(self, model, input_tensor, layers):
        """Measure FLOPs using thop"""
        if not HAS_THOP:
            return None, None

        # Create a wrapper to handle the forward pass with layers
        class ModelWrapper(nn.Module):
            def __init__(self, model, layers):
                super().__init__()
                self.model = model
                self.layers = layers

            def forward(self, x):
                return self.model(x, self.layers)

        wrapped_model = ModelWrapper(model, layers)

        try:
            flops, params = profile(
                wrapped_model, inputs=(input_tensor,), verbose=False
            )
            flops_formatted, params_formatted = clever_format([flops, params], "%.3f")
            return flops, flops_formatted
        except Exception as e:
            print(f"Error measuring FLOPs: {e}")
            return None, None

    def measure_memory(self, model, input_tensor, layers, num_iterations=10):
        """Measure peak GPU memory usage"""
        if not torch.cuda.is_available():
            return 0, 0

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        gc.collect()

        # Warmup
        with torch.no_grad():
            _ = model(input_tensor, layers)

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        # Measure during forward pass
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = model(input_tensor, layers)

        torch.cuda.synchronize()

        peak_memory = torch.cuda.max_memory_allocated() / (1024**3)  # Convert to GB
        current_memory = torch.cuda.memory_allocated() / (1024**3)

        return peak_memory, current_memory

    def measure_inference_time(self, model, input_tensor, layers, num_iterations=50):
        """Measure average inference time"""
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(input_tensor, layers)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Measure
        times = []
        with torch.no_grad():
            for _ in range(num_iterations):
                start = time.perf_counter()
                _ = model(input_tensor, layers)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                end = time.perf_counter()
                times.append((end - start) * 1000)  # Convert to ms

        avg_time = sum(times) / len(times)
        std_time = (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5

        return avg_time, std_time

    def analyze_architecture(self, model_name, config):
        """Analyze a single architecture"""
        print(f"\n{'='*60}")
        print(f"Analyzing {model_name.upper()}")
        print(f"{'='*60}")

        # Create model
        model = NeuralStyleTransfer(
            model_name=model_name,
            pooling=config.get("pooling", "ori"),
            device=self.device,
        )
        model.eval()

        # Get layers configuration
        content_layers = config["content_layers"]
        style_layers = config["style_layers"]

        # Create dummy input
        dummy_input = torch.randn(1, 3, self.image_size, self.image_size).to(
            self.device
        )

        # Count parameters
        total_params, trainable_params = self.count_parameters(model)
        print(f"Total Parameters: {total_params:,} ({total_params/1e6:.2f}M)")
        print(f"Trainable Parameters: {trainable_params:,}")

        # Measure FLOPs
        flops_raw, flops_formatted = self.measure_flops(
            model, dummy_input, content_layers
        )
        if flops_raw:
            print(f"FLOPs: {flops_formatted} ({flops_raw/1e9:.2f} GFLOPs)")
        else:
            print("FLOPs: N/A (thop not available)")

        # Measure memory
        peak_memory, current_memory = self.measure_memory(
            model, dummy_input, content_layers
        )
        print(f"Peak GPU Memory: {peak_memory:.3f} GB")
        print(f"Current GPU Memory: {current_memory:.3f} GB")

        # Measure inference time
        avg_time, std_time = self.measure_inference_time(
            model, dummy_input, content_layers
        )
        print(f"Avg Inference Time: {avg_time:.2f} ± {std_time:.2f} ms")

        # Store results
        result = {
            "Architecture": model_name.upper(),
            "Parameters (M)": total_params / 1e6,
            "FLOPs (G)": flops_raw / 1e9 if flops_raw else None,
            "FLOPs_formatted": flops_formatted if flops_formatted else "N/A",
            "Peak VRAM (GB)": peak_memory,
            "Avg Inference Time (ms)": avg_time,
            "Std Inference Time (ms)": std_time,
            "Content Layers": str(content_layers),
            "Style Layers": str(style_layers),
        }

        self.results.append(result)

        # Cleanup
        del model
        torch.cuda.empty_cache()
        gc.collect()

        return result

    def run_all_analyses(self, config_path="config_final.json"):
        """Run analysis for all architectures in config"""
        # Load config
        with open(config_path, "r") as f:
            config = json.load(f)

        models_config = config["models"]

        # Analyze each model
        for model_name, model_config in models_config.items():
            try:
                self.analyze_architecture(model_name, model_config)
            except Exception as e:
                print(f"Error analyzing {model_name}: {e}")
                import traceback

                traceback.print_exc()

        return self.results

    def generate_comparison_table(
        self, output_path="outputs/computational_analysis.csv"
    ):
        """Generate comparison table"""
        df = pd.DataFrame(self.results)

        # Reorder columns for better readability
        column_order = [
            "Architecture",
            "Parameters (M)",
            "FLOPs (G)",
            "Peak VRAM (GB)",
            "Avg Inference Time (ms)",
            "Std Inference Time (ms)",
        ]

        df_display = df[column_order].copy()

        # Round for display
        df_display["Parameters (M)"] = df_display["Parameters (M)"].round(2)
        if df_display["FLOPs (G)"].notna().any():
            df_display["FLOPs (G)"] = df_display["FLOPs (G)"].round(2)
        df_display["Peak VRAM (GB)"] = df_display["Peak VRAM (GB)"].round(3)
        df_display["Avg Inference Time (ms)"] = df_display[
            "Avg Inference Time (ms)"
        ].round(2)
        df_display["Std Inference Time (ms)"] = df_display[
            "Std Inference Time (ms)"
        ].round(2)

        # Save to CSV
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"\n{'='*60}")
        print(f"Results saved to: {output_path}")
        print(f"{'='*60}")

        # Print formatted table
        print("\n" + "=" * 80)
        print("COMPUTATIONAL COST COMPARISON")
        print("=" * 80)
        print(df_display.to_string(index=False))
        print("=" * 80)

        return df_display


def main():
    """Main execution"""
    print("=" * 80)
    print("NEURAL STYLE TRANSFER - COMPUTATIONAL COST ANALYSIS")
    print("=" * 80)

    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"\nGPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    else:
        print("\nWarning: CUDA not available, running on CPU")

    # Check thop availability
    if not HAS_THOP:
        print("\nInstalling thop for FLOPs calculation...")
        print("Run: pip install thop")

    # Create analyzer
    analyzer = ComputationalAnalyzer(image_size=512, device="cuda")

    # Run analyses
    results = analyzer.run_all_analyses("config_final.json")

    # Generate outputs
    analyzer.generate_comparison_table("outputs/computational_analysis.csv")

    # Save detailed results as JSON
    output_json = "outputs/computational_analysis.json"
    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to: {output_json}")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
