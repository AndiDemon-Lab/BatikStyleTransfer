import os
import json
import torch
import time
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import traceback

from src.models import NeuralStyleTransfer
from src.nst_utils import ImageHandler
from src.criterion import Criterion
from src.metrics import MetricsCalculator, LossTracker, Timer
from src.data_validation import TrainRequest
from torch import optim


class BatchExperimentRunner:
    """
    Run batch NST experiments with comprehensive metrics tracking.
    """

    def __init__(self, config):
        """
        Initialize batch runner with configuration.

        Args:
            config: Dictionary with experiment configuration
        """
        self.config = config
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Create output directories
        self.base_output_dir = Path(
            config.get("output_dir", "outputs/batch_experiments")
        )
        self.base_output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize handlers
        self.image_handler = ImageHandler()
        self.metrics_calculator = MetricsCalculator(device=self.device)

        # Experiment tracking
        self.experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results = []

    def get_image_files(self, directory, extensions=(".jpg", ".jpeg", ".png")):
        """
        Get all image files from a directory.

        Args:
            directory: Path to directory
            extensions: Tuple of valid extensions

        Returns:
            List of image file paths
        """
        directory = Path(directory)
        images = []
        for ext in extensions:
            images.extend(directory.glob(f"*{ext}"))
            images.extend(directory.glob(f"*{ext.upper()}"))
        return sorted(images)

    def run_single_experiment(
        self, model_name, content_path, style_path, experiment_idx, total_experiments
    ):
        """
        Run a single NST experiment.

        Args:
            model_name: Name of the model to use
            content_path: Path to content image
            style_path: Path to style image
            experiment_idx: Current experiment index
            total_experiments: Total number of experiments

        Returns:
            Dictionary with experiment results
        """
        # Create experiment-specific output directory
        content_name = Path(content_path).stem
        style_name = Path(style_path).stem
        exp_dir = (
            self.base_output_dir / f"{model_name}" / f"{content_name}_{style_name}"
        )
        exp_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*80}")
        print(f"Experiment {experiment_idx}/{total_experiments}")
        print(f"Model: {model_name} | Content: {content_name} | Style: {style_name}")
        print(f"{'='*80}")

        # Get model-specific configuration
        model_config = self.config["models"][model_name]

        # Initialize model
        nst = NeuralStyleTransfer(
            model_name,
            pretrained_weights_path=model_config.get("pretrained_weights_path"),
            pooling=model_config.get("pooling", "ori"),
            device=self.device,
        ).to(self.device)

        # Initialize criterion
        criterion = Criterion(
            content_weight=model_config.get("content_weight", 1),
            style_weight=model_config.get("style_weight", 1e8),
        )

        # Initialize loss tracker
        loss_tracker = LossTracker()

        # Load images
        content_image = self.image_handler.load_image(
            str(content_path), self.image_handler.transform
        ).to(self.device)

        style_image = self.image_handler.load_image(
            str(style_path), self.image_handler.transform
        ).to(self.device)

        # Prepare output image
        output = content_image.clone().to(self.device)
        output.requires_grad = True

        # Optimizer
        optimizer = optim.AdamW([output], lr=model_config.get("learning_rate", 0.05))

        # Extract features
        content_layers = model_config.get("content_layers", [[2, -1]])
        style_layers = model_config.get("style_layers", [[2, -1]])

        content_features = nst(content_image, layers=content_layers)
        style_features = nst(style_image, layers=style_layers)

        # Training parameters
        max_epochs = self.config.get("max_epochs", 5000)
        save_epochs = self.config.get(
            "save_epochs",
            [100, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000],
        )
        metric_epochs = self.config.get(
            "metric_epochs", 100
        )  # Calculate metrics every N epochs

        # Training loop
        total_timer = Timer()
        total_timer.start()

        generated_images = []

        with tqdm(total=max_epochs, desc=f"{model_name}", ncols=100) as pbar:
            for epoch in range(1, max_epochs + 1):
                epoch_timer = Timer()
                epoch_timer.start()

                # Forward pass
                output_content_features = nst(output, layers=content_layers)
                output_style_features = nst(output, layers=style_layers)

                # Calculate loss
                total_loss, content_loss, style_loss = criterion.criterion(
                    content_features,
                    style_features,
                    output_content_features,
                    output_style_features,
                )

                # Backward pass
                total_loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                # Track epoch time
                epoch_time = epoch_timer.stop()
                loss_tracker.add_epoch_time(epoch, epoch_time)

                # Track loss with all components
                loss_tracker.add_loss(
                    epoch, content_loss.item(), style_loss.item(), total_loss.item()
                )

                # Calculate metrics at specified intervals
                if epoch % metric_epochs == 0:
                    with torch.no_grad():
                        metrics = self.metrics_calculator.calculate_all_metrics(
                            output, content_image, style_image
                        )
                        loss_tracker.add_metrics(epoch, metrics)

                    pbar.set_postfix(
                        {
                            "Loss": f"{total_loss.item():.2e}",
                            "SSIM": f'{metrics.get("ssim", 0):.3f}',
                            "PSNR": f'{metrics.get("psnr", 0):.2f}',
                        }
                    )

                # Save output at specified epochs
                if epoch in save_epochs:
                    output_path = exp_dir / f"epoch_{epoch}.png"
                    self.image_handler.save_image(output, str(output_path))
                    generated_images.append({"epoch": epoch, "path": str(output_path)})

                pbar.update(1)

        total_time = total_timer.stop()

        # Final metrics
        with torch.no_grad():
            final_metrics = self.metrics_calculator.calculate_all_metrics(
                output, content_image, style_image
            )

        # Save final output
        final_output_path = exp_dir / "final_output.png"
        self.image_handler.save_image(output, str(final_output_path))

        # Compile experiment results
        experiment_result = {
            "experiment_id": self.experiment_id,
            "timestamp": datetime.now().isoformat(),
            "model_name": model_name,
            "content_image": str(content_path),
            "style_image": str(style_path),
            "content_name": content_name,
            "style_name": style_name,
            "hyperparameters": {
                "content_weight": model_config.get("content_weight", 1),
                "style_weight": model_config.get("style_weight", 1e8),
                "learning_rate": model_config.get("learning_rate", 0.05),
                "max_epochs": max_epochs,
                "content_layers": content_layers,
                "style_layers": style_layers,
                "pooling": model_config.get("pooling", "ori"),
            },
            "training_time_seconds": total_time,
            "final_metrics": final_metrics,
            "loss_tracking": loss_tracker.to_dict(),
            "generated_images": generated_images,
            "final_output": str(final_output_path),
            "output_directory": str(exp_dir),
        }

        # Save experiment metadata
        metadata_path = exp_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(experiment_result, f, indent=2)

        print(f"\n✓ Experiment completed in {total_time:.2f}s")
        print(f"  Final SSIM: {final_metrics.get('ssim', 0):.4f}")
        print(f"  Final PSNR: {final_metrics.get('psnr', 0):.2f} dB")
        if "lpips" in final_metrics:
            print(f"  Final LPIPS: {final_metrics.get('lpips', 0):.4f}")

        return experiment_result

    def run_batch(self):
        """
        Run all experiments in the batch.

        Returns:
            List of experiment results
        """
        # Get content and style images
        content_dir = self.config["content_dir"]
        style_dir = self.config["style_dir"]

        content_images = self.get_image_files(content_dir)
        style_images = self.get_image_files(style_dir)

        # Apply subset configuration if enabled
        subset_config = self.config.get("subset_config", {})
        if subset_config.get("enabled", False):
            num_content = subset_config.get("num_content", len(content_images))
            num_style = subset_config.get("num_style", len(style_images))
            subset_models = subset_config.get(
                "models", list(self.config["models"].keys())
            )

            content_images = content_images[:num_content]
            style_images = style_images[:num_style]

            # Filter models
            models_to_run = {
                k: v for k, v in self.config["models"].items() if k in subset_models
            }
        else:
            models_to_run = self.config["models"]

        print(f"\n{'='*80}")
        print(f"BATCH EXPERIMENT CONFIGURATION")
        print(f"{'='*80}")
        print(f"Content images: {len(content_images)}")
        print(f"Style images: {len(style_images)}")
        print(f"Models: {list(models_to_run.keys())}")
        print(
            f"Total experiments: {len(content_images) * len(style_images) * len(models_to_run)}"
        )
        print(f"Output directory: {self.base_output_dir}")
        if subset_config.get("enabled", False):
            print(f"SUBSET MODE ENABLED")
        print(f"{'='*80}\n")

        # Calculate total experiments
        total_experiments = len(content_images) * len(style_images) * len(models_to_run)
        experiment_idx = 0

        # Run experiments
        for model_name in models_to_run.keys():
            for content_path in content_images:
                for style_path in style_images:
                    experiment_idx += 1

                    try:
                        result = self.run_single_experiment(
                            model_name,
                            content_path,
                            style_path,
                            experiment_idx,
                            total_experiments,
                        )
                        self.results.append(result)

                    except Exception as e:
                        print(f"\n✗ Error in experiment {experiment_idx}:")
                        print(f"  Model: {model_name}")
                        print(f"  Content: {content_path}")
                        print(f"  Style: {style_path}")
                        print(f"  Error: {str(e)}")
                        traceback.print_exc()

                        # Log error
                        error_result = {
                            "experiment_id": self.experiment_id,
                            "timestamp": datetime.now().isoformat(),
                            "model_name": model_name,
                            "content_image": str(content_path),
                            "style_image": str(style_path),
                            "status": "failed",
                            "error": str(e),
                            "traceback": traceback.format_exc(),
                        }
                        self.results.append(error_result)

                        # Continue with next experiment
                        continue

        # Save aggregated results
        self.save_aggregated_results()

        return self.results

    def save_aggregated_results(self):
        """
        Save aggregated results from all experiments.
        """
        aggregated_path = (
            self.base_output_dir / f"aggregated_results_{self.experiment_id}.json"
        )

        aggregated_data = {
            "experiment_id": self.experiment_id,
            "timestamp": datetime.now().isoformat(),
            "configuration": self.config,
            "total_experiments": len(self.results),
            "successful_experiments": len(
                [r for r in self.results if r.get("status") != "failed"]
            ),
            "failed_experiments": len(
                [r for r in self.results if r.get("status") == "failed"]
            ),
            "results": self.results,
        }

        with open(aggregated_path, "w") as f:
            json.dump(aggregated_data, f, indent=2)

        print(f"\n{'='*80}")
        print(f"BATCH EXPERIMENT COMPLETED")
        print(f"{'='*80}")
        print(f"Total experiments: {len(self.results)}")
        print(f"Successful: {aggregated_data['successful_experiments']}")
        print(f"Failed: {aggregated_data['failed_experiments']}")
        print(f"Results saved to: {aggregated_path}")
        print(f"{'='*80}\n")


def create_default_config():
    """
    Create default configuration for batch experiments.

    Returns:
        Dictionary with default configuration
    """
    return {
        "content_dir": "data/_content",
        "style_dir": "data/_style",
        "output_dir": "outputs/batch_experiments",
        "max_epochs": 5000,
        "save_epochs": [100, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000],
        "metric_epochs": 100,
        "models": {
            "vgg19": {
                "content_weight": 1,
                "style_weight": 1e8,
                "learning_rate": 0.05,
                "pooling": "ori",
                "content_layers": [2],
                "style_layers": [8],
                "pretrained_weights_path": None,
            },
            "vgg16": {
                "content_weight": 1,
                "style_weight": 1e8,
                "learning_rate": 0.05,
                "pooling": "ori",
                "content_layers": [2],
                "style_layers": [8],
                "pretrained_weights_path": None,
            },
            "resnet50": {
                "content_weight": 1,
                "style_weight": 1e8,
                "learning_rate": 0.05,
                "pooling": "ori",
                "content_layers": [[2, -1]],
                "style_layers": [[2, -1]],
                "pretrained_weights_path": None,
            },
            "resnet101": {
                "content_weight": 1,
                "style_weight": 1e8,
                "learning_rate": 0.05,
                "pooling": "ori",
                "content_layers": [[2, -1]],
                "style_layers": [[2, -1]],
                "pretrained_weights_path": None,
            },
            "inception_v3": {
                "content_weight": 1,
                "style_weight": 1e8,
                "learning_rate": 0.05,
                "pooling": "ori",
                "content_layers": [4],
                "style_layers": [8],
                "pretrained_weights_path": None,
            },
        },
    }


if __name__ == "__main__":
    # Create configuration
    config = create_default_config()

    # Option: Load configuration from JSON file
    # with open('experiment_config.json', 'r') as f:
    #     config = json.load(f)

    # Create and run batch experiment
    runner = BatchExperimentRunner(config)
    results = runner.run_batch()

    print("\n✓ All experiments completed!")
