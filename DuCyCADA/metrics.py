"""
metrics.py
----------
Image quality metrics and helper loss utilities for DuCyCADA evaluation.

Contents:
    - normalize_tensor        : Min-max normalize a tensor to [0, 1].
    - scale_tensor            : Scale a [0,1] tensor to an arbitrary range.
    - show_difference         : Compute absolute pixel-difference map.
    - histogram_matching      : Match image histogram to a reference.
    - _filter                 : Gaussian kernel for VIF computation.
    - _vif_per_channel        : Per-channel VIF helper.
    - visual_information_fidelity : Multi-scale VIF metric (pixel-based).
    - VisualInformationFidelity   : Stateful nn.Module wrapper for VIF.
    - VIF                     : Convenience function wrapper.

Note
----
The multi-scale ``visual_information_fidelity`` and ``_vif_per_channel``
implementations are adapted from the torchmetrics / piq libraries
(Apache 2.0 License, Copyright The PyTorch Lightning team).
"""

from typing import Any

import cv2
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
from torch.nn.functional import conv2d
from skimage.exposure import match_histograms


# ---------------------------------------------------------------------------
# Tensor normalization helpers
# ---------------------------------------------------------------------------

def normalize_tensor(tensor: Tensor) -> Tensor:
    """Min-max normalize a tensor to the range [0, 1].

    Args:
        tensor: Input tensor of any shape.

    Returns:
        Tensor normalized to [0, 1].
    """
    min_val = tensor.min()
    max_val = tensor.max()
    return (tensor - min_val) / (max_val - min_val + 1e-8)


def scale_tensor(tensor: Tensor, new_min: float, new_max: float) -> Tensor:
    """Linearly scale a [0, 1] tensor to [new_min, new_max].

    Args:
        tensor:  Input tensor, expected in [0, 1].
        new_min: Lower bound of the output range.
        new_max: Upper bound of the output range.

    Returns:
        Rescaled tensor.
    """
    return tensor * (new_max - new_min) + new_min


# ---------------------------------------------------------------------------
# Pixel-level difference visualization
# ---------------------------------------------------------------------------

def show_difference(image1: np.ndarray, image2: np.ndarray) -> np.ndarray:
    """Compute and normalize the absolute pixel-level difference between two images.

    Args:
        image1: First image as a NumPy array (H, W) or (H, W, C).
        image2: Second image as a NumPy array with the same shape as image1.

    Returns:
        Normalized absolute difference map in [0, 1] as float32, or None
        if the inputs are incompatible.
    """
    if image1 is None or image2 is None:
        print("Error: One or both images are None.")
        return None

    if image1.shape != image2.shape:
        print(f"Error: Shape mismatch — {image1.shape} vs {image2.shape}.")
        return None

    difference = cv2.absdiff(image1.astype(np.float32), image2.astype(np.float32))
    normalized = cv2.normalize(
        difference, None, alpha=0, beta=1,
        norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
    )
    return normalized.astype(np.float32)


# ---------------------------------------------------------------------------
# Histogram matching
# ---------------------------------------------------------------------------

def histogram_matching(reference: np.ndarray, image: np.ndarray) -> np.ndarray:
    """Match the histogram of *image* to that of *reference*.

    Args:
        reference: Reference image whose histogram is the target.
        image:     Image to be transformed.

    Returns:
        Histogram-matched version of *image*.
    """
    return match_histograms(image, reference)


# ---------------------------------------------------------------------------
# Visual Information Fidelity (VIF) — multi-scale pixel-based implementation
# Adapted from torchmetrics / piq (Apache 2.0)
# ---------------------------------------------------------------------------

def _filter(win_size: float, sigma: float,
            dtype: torch.dtype, device: torch.device) -> Tensor:
    """Build a 2-D Gaussian kernel.

    Args:
        win_size: Kernel window size (side length).
        sigma:    Standard deviation of the Gaussian.
        dtype:    Desired tensor dtype.
        device:   Desired tensor device.

    Returns:
        Normalized 2-D Gaussian kernel of shape (win_size, win_size).
    """
    coords = torch.arange(win_size, dtype=dtype, device=device) - (win_size - 1) / 2
    g = coords ** 2
    g = torch.exp(-(g.unsqueeze(0) + g.unsqueeze(1)) / (2.0 * sigma ** 2))
    g /= torch.sum(g)
    return g


def _vif_per_channel(preds: Tensor, target: Tensor, sigma_n_sq: float) -> Tensor:
    """Compute the multi-scale VIF score for a single image channel.

    Args:
        preds:      Predicted image channel of shape (N, H, W).
        target:     Ground-truth image channel of shape (N, H, W).
        sigma_n_sq: Variance of the assumed visual noise model.

    Returns:
        Per-image VIF score tensor of shape (N,).
    """
    dtype  = preds.dtype
    device = preds.device

    # Add explicit channel dimension for conv2d
    preds  = preds.unsqueeze(1)
    target = target.unsqueeze(1)

    eps        = torch.tensor(1e-10, dtype=dtype, device=device)
    sigma_n_sq = torch.tensor(sigma_n_sq, dtype=dtype, device=device)

    preds_vif  = torch.zeros(1, dtype=dtype, device=device)
    target_vif = torch.zeros(1, dtype=dtype, device=device)

    for scale in range(4):
        n      = 2.0 ** (4 - scale) + 1
        kernel = _filter(n, n / 5, dtype=dtype, device=device)[None, None, :]

        # Downsample at every scale after the first
        if scale > 0:
            target = conv2d(target, kernel)[:, :, ::2, ::2]
            preds  = conv2d(preds,  kernel)[:, :, ::2, ::2]

        # Local means
        mu_target = conv2d(target, kernel)
        mu_preds  = conv2d(preds,  kernel)

        # Local variances and covariance
        sigma_target_sq = torch.clamp(conv2d(target ** 2, kernel) - mu_target ** 2, min=0.0)
        sigma_preds_sq  = torch.clamp(conv2d(preds  ** 2, kernel) - mu_preds  ** 2, min=0.0)
        sigma_target_preds = conv2d(target * preds, kernel) - mu_target * mu_preds

        # Gain and distortion variance
        g          = sigma_target_preds / (sigma_target_sq + eps)
        sigma_v_sq = sigma_preds_sq - g * sigma_target_preds

        # Handle edge cases
        mask = sigma_target_sq < eps
        g[mask], sigma_v_sq[mask], sigma_target_sq[mask] = 0, sigma_preds_sq[mask], 0

        mask = sigma_preds_sq < eps
        g[mask], sigma_v_sq[mask] = 0, 0

        mask = g < 0
        sigma_v_sq[mask], g[mask] = sigma_preds_sq[mask], 0
        sigma_v_sq = torch.clamp(sigma_v_sq, min=eps)

        # Accumulate VIF numerator and denominator
        preds_vif  = preds_vif  + torch.sum(
            torch.log10(1.0 + (g ** 2.0) * sigma_target_sq / (sigma_v_sq + sigma_n_sq)),
            dim=[1, 2, 3]
        )
        target_vif = target_vif + torch.sum(
            torch.log10(1.0 + sigma_target_sq / sigma_n_sq),
            dim=[1, 2, 3]
        )

    return preds_vif / target_vif


def visual_information_fidelity(preds: Tensor, target: Tensor,
                                sigma_n_sq: float = 2.0) -> Tensor:
    """Compute pixel-based Visual Information Fidelity (VIF-P).

    Multi-scale implementation inspired by torchmetrics / piq.

    Args:
        preds:      Predicted images of shape (N, C, H, W). H and W >= 41.
        target:     Ground-truth images of shape (N, C, H, W). H and W >= 41.
        sigma_n_sq: Variance of the visual noise model (default: 2.0).

    Returns:
        Scalar tensor with the mean VIF score across the batch and channels.

    Raises:
        ValueError: If spatial dimensions are smaller than 41×41.
    """
    if preds.size(-1) < 41 or preds.size(-2) < 41:
        raise ValueError(
            f"Input too small for VIF. Expected >= 41×41, got "
            f"{preds.size(-2)}×{preds.size(-1)}."
        )
    if target.size(-1) < 41 or target.size(-2) < 41:
        raise ValueError(
            f"Target too small for VIF. Expected >= 41×41, got "
            f"{target.size(-2)}×{target.size(-1)}."
        )

    per_channel = [
        _vif_per_channel(preds[:, i, :, :], target[:, i, :, :], sigma_n_sq)
        for i in range(preds.size(1))
    ]
    # Average across channels, then across batch
    return torch.cat(per_channel).mean()


class VisualInformationFidelity(nn.Module):
    """Stateful nn.Module for accumulating VIF across multiple batches.

    Useful when computing an aggregate metric over a full test set.

    Args:
        sigma_n_sq: Variance of the visual noise model (default: 2.0).

    Example::

        vif = VisualInformationFidelity()
        for preds, target in dataloader:
            vif.update(preds, target)
        score = vif.compute()
    """

    def __init__(self, sigma_n_sq: float = 2.0, **kwargs: Any) -> None:
        super().__init__()

        if not isinstance(sigma_n_sq, (float, int)) or sigma_n_sq < 0:
            raise ValueError(
                f"`sigma_n_sq` must be a non-negative float, got {sigma_n_sq}."
            )

        self.sigma_n_sq = sigma_n_sq
        self.vif_score  = torch.tensor(0.0)
        self.total      = torch.tensor(0.0)

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Accumulate VIF scores over a batch.

        Args:
            preds:  Predicted image tensor (N, C, H, W).
            target: Ground-truth image tensor (N, C, H, W).
        """
        channels = preds.size(1)
        per_ch = [
            self._vif_per_channel(preds[:, i, :, :], target[:, i, :, :])
            for i in range(channels)
        ]
        score = torch.mean(torch.stack(per_ch), 0) if channels > 1 else torch.cat(per_ch)
        self.vif_score += torch.sum(score)
        self.total     += preds.shape[0]

    def _vif_per_channel(self, pred: Tensor, target: Tensor) -> Tensor:
        """Simplified single-scale VIF for the stateful module.

        Args:
            pred:   Predicted channel tensor (N, H, W).
            target: Ground-truth channel tensor (N, H, W).

        Returns:
            VIF score tensor.
        """
        mu_pred   = torch.mean(pred)
        mu_target = torch.mean(target)

        sigma_pred_sq   = torch.var(pred)
        sigma_target_sq = torch.var(target)
        covariance      = torch.mean((pred - mu_pred) * (target - mu_target))

        g          = covariance / (sigma_pred_sq + 1e-10)
        sigma_v_sq = sigma_target_sq - g * covariance

        return torch.log2(
            1 + (g ** 2 * sigma_pred_sq) / (sigma_v_sq + self.sigma_n_sq + 1e-10)
        )

    def compute(self) -> Tensor:
        """Return the accumulated mean VIF score.

        Returns:
            Scalar tensor.
        """
        return self.vif_score / self.total


def VIF(preds: Tensor, target: Tensor) -> Tensor:
    """Convenience wrapper: compute multi-scale VIF for a single batch.

    Args:
        preds:  Predicted image tensor (N, C, H, W).
        target: Ground-truth image tensor (N, C, H, W).

    Returns:
        Scalar VIF score.
    """
    return visual_information_fidelity(preds, target)
