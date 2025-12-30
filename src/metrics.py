"""
metrics module for Neural Style Transfer experimental evaluation."""

import torch
import torch.nn as nn
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr_skimage
from PIL import Image
import time

try:
    import lpips

    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("Warning: LPIPS not available. Install with: pip install lpips")


class MetricsCalculator:
    """
    Comprehensive metrics calculator for NST evaluation.

    Metrics included:
    - SSIM (Structural Similarity Index)
    - PSNR (Peak Signal-to-Noise Ratio)
    - LPIPS (Learned Perceptual Image Patch Similarity)
    - MSE (Mean Squared Error)
    - Content Loss (per layer)
    - Style Loss (per layer)
    """

    def __init__(self, device="cpu"):
        self.device = device
        self.mean = torch.FloatTensor([[[0.485, 0.456, 0.406]]]).to(device)
        self.std = torch.FloatTensor([[[0.229, 0.224, 0.225]]]).to(device)

        # Initialize LPIPS model if available
        if LPIPS_AVAILABLE:
            self.lpips_model = lpips.LPIPS(net="vgg").to(device)
            self.lpips_model.eval()
        else:
            self.lpips_model = None

    def denormalize(self, tensor):
        """
        Denormalize tensor from ImageNet normalization.

        Args:
            tensor: Normalized tensor (C, H, W) or (B, C, H, W)

        Returns:
            Denormalized tensor in range [0, 1]
        """
        # Ensure tensor is 4D (B, C, H, W)
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)

        # Reshape mean and std to match tensor dimensions (1, 3, 1, 1)
        mean = self.mean.view(1, 3, 1, 1)
        std = self.std.view(1, 3, 1, 1)

        denorm = tensor * std + mean
        return denorm.clamp(0, 1)

    def to_numpy_image(self, tensor):
        """
        Convert tensor to numpy image for skimage metrics.

        Args:
            tensor: Tensor (B, C, H, W) or (C, H, W)

        Returns:
            Numpy array (H, W, C) in range [0, 255]
        """
        # Denormalize (this will make it 4D)
        img = self.denormalize(tensor)

        # Squeeze to remove batch dimension
        if img.dim() == 4:
            img = img.squeeze(0)

        # Convert to numpy
        img = img.cpu().detach().numpy()
        img = img.transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)
        img = (img * 255).astype(np.uint8)

        return img

    def calculate_ssim(self, img1, img2, win_size=11):
        """
        Calculate Structural Similarity Index (SSIM).

        Args:
            img1: First image tensor
            img2: Second image tensor
            win_size: Window size for SSIM calculation

        Returns:
            SSIM value (float)
        """
        img1_np = self.to_numpy_image(img1)
        img2_np = self.to_numpy_image(img2)

        ssim_value, _ = ssim(
            img1_np,
            img2_np,
            win_size=win_size,
            channel_axis=2,
            full=True,
            data_range=255,
        )

        return float(ssim_value)

    def calculate_psnr(self, img1, img2):
        """
        Calculate Peak Signal-to-Noise Ratio (PSNR).

        Args:
            img1: First image tensor
            img2: Second image tensor

        Returns:
            PSNR value in dB (float)
        """
        img1_np = self.to_numpy_image(img1)
        img2_np = self.to_numpy_image(img2)

        psnr_value = psnr_skimage(img1_np, img2_np, data_range=255)

        return float(psnr_value)

    def calculate_mse(self, img1, img2):
        """
        Calculate Mean Squared Error (MSE).

        Args:
            img1: First image tensor
            img2: Second image tensor

        Returns:
            MSE value (float)
        """
        # Denormalize first
        img1_denorm = self.denormalize(img1)
        img2_denorm = self.denormalize(img2)

        mse = torch.mean((img1_denorm - img2_denorm) ** 2)

        return float(mse.item())

    def calculate_lpips(self, img1, img2):
        """
        Calculate Learned Perceptual Image Patch Similarity (LPIPS).
        Lower values indicate better perceptual similarity.

        Args:
            img1: First image tensor
            img2: Second image tensor

        Returns:
            LPIPS value (float), or None if LPIPS not available
        """
        if self.lpips_model is None:
            return None

        # LPIPS expects images in range [-1, 1]
        img1_denorm = self.denormalize(img1)
        img2_denorm = self.denormalize(img2)

        # Convert [0, 1] to [-1, 1]
        img1_lpips = img1_denorm * 2 - 1
        img2_lpips = img2_denorm * 2 - 1

        with torch.no_grad():
            lpips_value = self.lpips_model(img1_lpips, img2_lpips)

        return float(lpips_value.item())

    def calculate_all_metrics(self, output, content, style=None):
        """
        Calculate all available metrics.

        Args:
            output: Generated output image
            content: Original content image
            style: Style image (optional, for style-specific metrics)

        Returns:
            Dictionary of metrics
        """
        metrics = {}

        # Content preservation metrics
        metrics["ssim"] = self.calculate_ssim(output, content)
        metrics["psnr"] = self.calculate_psnr(output, content)
        metrics["mse"] = self.calculate_mse(output, content)

        # Perceptual metrics
        lpips_value = self.calculate_lpips(output, content)
        if lpips_value is not None:
            metrics["lpips"] = lpips_value

        return metrics


class LossTracker:
    """
    Track loss values during training for analysis.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all tracked values."""
        self.content_losses = []
        self.style_losses = []
        self.total_losses = []
        self.content_losses_per_layer = {}
        self.style_losses_per_layer = {}
        self.epoch_times = []
        self.metrics_history = {"ssim": [], "psnr": [], "mse": [], "lpips": []}

    def add_loss(self, epoch, content_loss, style_loss, total_loss):
        """
        Add loss values for an epoch.

        Args:
            epoch: Current epoch number
            content_loss: Content loss value
            style_loss: Style loss value
            total_loss: Total loss value
        """
        self.content_losses.append({"epoch": epoch, "value": float(content_loss)})
        self.style_losses.append({"epoch": epoch, "value": float(style_loss)})
        self.total_losses.append({"epoch": epoch, "value": float(total_loss)})

    def add_layer_loss(self, epoch, layer_name, content_loss, style_loss):
        """
        Add per-layer loss values.

        Args:
            epoch: Current epoch number
            layer_name: Name/index of the layer
            content_loss: Content loss for this layer
            style_loss: Style loss for this layer
        """
        if layer_name not in self.content_losses_per_layer:
            self.content_losses_per_layer[layer_name] = []
            self.style_losses_per_layer[layer_name] = []

        self.content_losses_per_layer[layer_name].append(
            {"epoch": epoch, "value": float(content_loss)}
        )
        self.style_losses_per_layer[layer_name].append(
            {"epoch": epoch, "value": float(style_loss)}
        )

    def add_metrics(self, epoch, metrics):
        """
        Add metrics for an epoch.

        Args:
            epoch: Current epoch number
            metrics: Dictionary of metric values
        """
        for metric_name, value in metrics.items():
            if metric_name in self.metrics_history:
                self.metrics_history[metric_name].append(
                    {"epoch": epoch, "value": float(value)}
                )

    def add_epoch_time(self, epoch, time_seconds):
        """
        Add time taken for an epoch.

        Args:
            epoch: Current epoch number
            time_seconds: Time in seconds
        """
        self.epoch_times.append({"epoch": epoch, "time": float(time_seconds)})

    def get_summary(self):
        """
        Get summary statistics of tracked values.

        Returns:
            Dictionary with summary statistics
        """
        summary = {}

        # Loss statistics
        if self.total_losses:
            total_loss_values = [x["value"] for x in self.total_losses]
            summary["loss"] = {
                "final": total_loss_values[-1],
                "min": min(total_loss_values),
                "max": max(total_loss_values),
                "mean": np.mean(total_loss_values),
                "std": np.std(total_loss_values),
            }

        # Metrics statistics
        for metric_name, history in self.metrics_history.items():
            if history:
                values = [x["value"] for x in history]
                summary[metric_name] = {
                    "final": values[-1],
                    "min": min(values),
                    "max": max(values),
                    "mean": np.mean(values),
                    "std": np.std(values),
                }

        # Time statistics
        if self.epoch_times:
            times = [x["time"] for x in self.epoch_times]
            summary["time"] = {
                "total": sum(times),
                "mean_per_epoch": np.mean(times),
                "std_per_epoch": np.std(times),
            }

        return summary

    def to_dict(self):
        """
        Convert all tracked data to dictionary for JSON serialization.

        Returns:
            Dictionary with all tracked data
        """
        return {
            "content_losses": self.content_losses,
            "style_losses": self.style_losses,
            "total_losses": self.total_losses,
            "content_losses_per_layer": self.content_losses_per_layer,
            "style_losses_per_layer": self.style_losses_per_layer,
            "epoch_times": self.epoch_times,
            "metrics_history": self.metrics_history,
            "summary": self.get_summary(),
        }


class Timer:
    """Simple timer for measuring execution time."""

    def __init__(self):
        self.start_time = None
        self.elapsed = 0

    def start(self):
        """Start the timer."""
        self.start_time = time.time()

    def stop(self):
        """Stop the timer and return elapsed time."""
        if self.start_time is None:
            return 0
        self.elapsed = time.time() - self.start_time
        return self.elapsed

    def get_elapsed(self):
        """Get elapsed time without stopping."""
        if self.start_time is None:
            return 0
        return time.time() - self.start_time
