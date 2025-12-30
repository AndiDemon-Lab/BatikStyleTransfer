import json
from pathlib import Path
from batch_runner import BatchExperimentRunner


class PairedBatchRunner(BatchExperimentRunner):
    """
    Extended batch runner for paired content-style images.
    """

    def get_paired_images(self, base_dir):
        """
        Get paired content-style images from numbered directories.

        Args:
            base_dir: Base directory containing numbered folders

        Returns:
            List of tuples (content_path, style_path, pair_id)
        """
        base_path = Path(base_dir)
        pairs = []

        # Get all numbered directories
        for dir_path in sorted(base_path.iterdir()):
            if dir_path.is_dir() and dir_path.name.isdigit():
                content_path = dir_path / "content.jpg"
                style_path = dir_path / "style.jpg"

                if content_path.exists() and style_path.exists():
                    pairs.append((content_path, style_path, dir_path.name))
                else:
                    print(f"Warning: Missing content or style in {dir_path}")

        return pairs

    def run_single_experiment(
        self,
        model_name,
        content_path,
        style_path,
        experiment_idx,
        total_experiments,
        pair_id=None,
    ):
        """
        Override to use pair_id for folder naming.

        Args:
            model_name: Name of the model to use
            content_path: Path to content image
            style_path: Path to style image
            experiment_idx: Current experiment index
            total_experiments: Total number of experiments
            pair_id: Pair ID for folder naming (optional)

        Returns:
            Dictionary with experiment results
        """
        from pathlib import Path
        import torch
        from torch import optim
        from tqdm import tqdm
        from src.metrics import Timer
        from datetime import datetime
        import json

        # Create experiment-specific output directory using pair_id
        if pair_id:
            exp_dir = self.base_output_dir / f"{model_name}" / f"pair_{pair_id}"
        else:
            content_name = Path(content_path).stem
            style_name = Path(style_path).stem
            exp_dir = (
                self.base_output_dir / f"{model_name}" / f"{content_name}_{style_name}"
            )

        exp_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*80}")
        print(f"Experiment {experiment_idx}/{total_experiments}")
        print(f"Model: {model_name} | Pair ID: {pair_id if pair_id else 'N/A'}")
        print(f"{'='*80}")

        # Get model-specific configuration
        model_config = self.config["models"][model_name]

        # Initialize model
        from src.models import NeuralStyleTransfer

        nst = NeuralStyleTransfer(
            model_name,
            pretrained_weights_path=model_config.get("pretrained_weights_path"),
            pooling=model_config.get("pooling", "ori"),
            device=self.device,
        ).to(self.device)

        # Initialize criterion
        from src.criterion import Criterion

        criterion = Criterion(
            content_weight=model_config.get("content_weight", 1),
            style_weight=model_config.get("style_weight", 1e8),
        )

        # Initialize loss tracker
        from src.metrics import LossTracker

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
        metric_epochs = self.config.get("metric_epochs", 100)

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

                # Track all losses
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
            "pair_id": pair_id,
            "content_image": str(content_path),
            "style_image": str(style_path),
            "content_name": Path(content_path).stem,
            "style_name": Path(style_path).stem,
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

    def run_paired_batch(self):
        """
        Run batch experiments on paired data.

        Returns:
            List of experiment results
        """
        # Get paired images
        paired_dir = self.config["paired_dir"]
        pairs = self.get_paired_images(paired_dir)

        # Apply subset if configured
        subset_config = self.config.get("subset_config", {})
        if subset_config.get("enabled", False):
            # Filter by specific pair IDs if provided
            pair_ids = subset_config.get("pair_ids", [])
            if pair_ids:
                # Filter pairs to only include specified IDs
                # pairs structure: (content_path, style_path, pair_id)
                filtered_pairs = []
                for pair in pairs:
                    pair_id = pair[2]  # Third element is pair_id
                    if int(pair_id) in pair_ids:
                        filtered_pairs.append(pair)
                pairs = filtered_pairs
            else:
                # Fallback to num_pairs (first N pairs)
                num_pairs = subset_config.get("num_pairs", len(pairs))
                pairs = pairs[:num_pairs]

            # Filter models
            subset_models = subset_config.get(
                "models", list(self.config["models"].keys())
            )
            models_to_run = {
                k: v for k, v in self.config["models"].items() if k in subset_models
            }
        else:
            models_to_run = self.config["models"]

        print(f"\n{'='*80}")
        print(f"PAIRED BATCH EXPERIMENT CONFIGURATION")
        print(f"{'='*80}")
        print(f"Paired images: {len(pairs)}")
        print(f"Models: {list(models_to_run.keys())}")
        print(f"Total experiments: {len(pairs) * len(models_to_run)}")
        print(f"Output directory: {self.base_output_dir}")
        if subset_config.get("enabled", False):
            print(f"SUBSET MODE ENABLED")
        print(f"{'='*80}\n")

        # Calculate total experiments
        total_experiments = len(pairs) * len(models_to_run)
        experiment_idx = 0

        # Run experiments
        for model_name in models_to_run.keys():
            for content_path, style_path, pair_id in pairs:
                experiment_idx += 1

                # Resume logic: Skip if metadata exists
                if pair_id:
                    check_dir = (
                        self.base_output_dir / f"{model_name}" / f"pair_{pair_id}"
                    )
                    if (check_dir / "metadata.json").exists():
                        print(
                            f"✓ Skipping Exp {experiment_idx}/{total_experiments}: {model_name} - Pair {pair_id} (Already completed)"
                        )
                        continue

                try:
                    result = self.run_single_experiment(
                        model_name,
                        content_path,
                        style_path,
                        experiment_idx,
                        total_experiments,
                        pair_id=pair_id,
                    )
                    self.results.append(result)

                except Exception as e:
                    print(f"\n✗ Error in experiment {experiment_idx}:")
                    print(f"  Model: {model_name}")
                    print(f"  Pair ID: {pair_id}")
                    print(f"  Content: {content_path}")
                    print(f"  Style: {style_path}")
                    print(f"  Error: {str(e)}")

                    import traceback

                    traceback.print_exc()

                    # Log error
                    error_result = {
                        "experiment_id": self.experiment_id,
                        "timestamp": __import__("datetime").datetime.now().isoformat(),
                        "model_name": model_name,
                        "pair_id": pair_id,
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


if __name__ == "__main__":
    # Load configuration
    with open("config_full_batch.json", "r") as f:
        config = json.load(f)

    print("=" * 80)
    print("STARTING FULL BATCH EXPERIMENT WITH PAIRED DATA")
    print("=" * 80)
    print(f"Paired directory: {config['paired_dir']}")
    print(f"Models: {list(config['models'].keys())}")
    print(f"Max epochs: {config['max_epochs']}")
    print("=" * 80)

    # Create and run paired batch experiment
    runner = PairedBatchRunner(config)
    results = runner.run_paired_batch()

    print("\n✓ All paired experiments completed!")
    print(f"Results saved to: {runner.base_output_dir}")
