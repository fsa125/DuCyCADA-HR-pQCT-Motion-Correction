"""
utils.py
--------
Utility functions for DuCyCADA / SWINDyT.

Contains (all unchanged from original notebook):
    - apply_fixed_gaussian_blur()
    - normalize_tensor()
    - scale_tensor()
    - show_difference()
    - histogram_matching()
    - VIF / visual_information_fidelity helpers
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from skimage.exposure import match_histograms
from torch import Tensor
from torch.nn.functional import conv2d

from torchmetrics.utilities.distributed import reduce


# ---------------------------------------------------------------------------
# Image processing  (unchanged from notebook cells 16 / 28 / 38 / 39 / 42)
# ---------------------------------------------------------------------------

def apply_fixed_gaussian_blur(imgs, kernel_size=5, sigma=0.3):
    """Apply a fixed (non-learned) Gaussian blur via depthwise convolution."""

    def gaussian_1d_kernel(kernel_size, sigma):
        coords = torch.arange(kernel_size).float() - kernel_size // 2
        kernel = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        kernel = kernel / kernel.sum()
        return kernel

    g_1d = gaussian_1d_kernel(kernel_size, sigma)
    g_2d = torch.outer(g_1d, g_1d)
    g_2d = g_2d.expand(imgs.shape[1], 1, kernel_size, kernel_size).to(imgs.device)
    padding = kernel_size // 2
    blurred = F.conv2d(imgs, g_2d, padding=padding, groups=imgs.shape[1])
    return blurred


def normalize_tensor(tensor):
    """Min-max normalize a tensor to [0, 1]."""
    min_val = tensor.min()
    max_val = tensor.max()
    normalized_tensor = (tensor - min_val) / (max_val - min_val)
    return normalized_tensor


def scale_tensor(tensor, new_min, new_max):
    """Scale a [0,1]-normalized tensor to [new_min, new_max]."""
    scaled_tensor = tensor * (new_max - new_min) + new_min
    return scaled_tensor


def show_difference(image1, image2):
    """Compute and return a normalized absolute-difference map (float32)."""
    if image1 is None or image2 is None:
        print("Error: One or both images not found or unable to read.")
        return
    if image1.shape != image2.shape:
        print("Error: Images do not have the same dimensions.")
        return
    difference = cv2.absdiff(image1.astype(np.float32), image2.astype(np.float32))
    normalized_difference = cv2.normalize(
        difference, None, alpha=0, beta=1,
        norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F,
    )
    return normalized_difference.astype(np.float32)


def histogram_matching(reference, image):
    """Match the histogram of *image* to *reference* using skimage."""
    matched = match_histograms(image, reference)
    return matched


# ---------------------------------------------------------------------------
# VIF metric  (unchanged from notebook cells 50 / 57)
# Licensed under the Apache License, Version 2.0 by The PyTorch Lightning team.
# ---------------------------------------------------------------------------

def _filter(win_size: float, sigma: float, dtype: torch.dtype, device: torch.device) -> Tensor:
    coords = torch.arange(win_size, dtype=dtype, device=device) - (win_size - 1) / 2
    g = coords ** 2
    g = torch.exp(-(g.unsqueeze(0) + g.unsqueeze(1)) / (2.0 * sigma ** 2))
    g /= torch.sum(g)
    return g


def _vif_per_channel(preds: Tensor, target: Tensor, sigma_n_sq: float) -> Tensor:
    dtype  = preds.dtype
    device = preds.device

    preds  = preds.unsqueeze(1)
    target = target.unsqueeze(1)
    eps    = torch.tensor(1e-10, dtype=dtype, device=device)
    sigma_n_sq = torch.tensor(sigma_n_sq, dtype=dtype, device=device)

    preds_vif  = torch.zeros(1, dtype=dtype, device=device)
    target_vif = torch.zeros(1, dtype=dtype, device=device)

    for scale in range(4):
        n      = 2.0 ** (4 - scale) + 1
        kernel = _filter(n, n / 5, dtype=dtype, device=device)[None, None, :]

        if scale > 0:
            target = conv2d(target, kernel)[:, :, ::2, ::2]
            preds  = conv2d(preds,  kernel)[:, :, ::2, ::2]

        mu_target = conv2d(target, kernel)
        mu_preds  = conv2d(preds,  kernel)
        mu_target_sq    = mu_target ** 2
        mu_preds_sq     = mu_preds  ** 2
        mu_target_preds = mu_target  * mu_preds

        sigma_target_sq    = torch.clamp(conv2d(target ** 2, kernel) - mu_target_sq,  min=0.0)
        sigma_preds_sq     = torch.clamp(conv2d(preds  ** 2, kernel) - mu_preds_sq,   min=0.0)
        sigma_target_preds = conv2d(target * preds, kernel) - mu_target_preds

        g           = sigma_target_preds / (sigma_target_sq + eps)
        sigma_v_sq  = sigma_preds_sq - g * sigma_target_preds

        mask = sigma_target_sq < eps
        g[mask]           = 0
        sigma_v_sq[mask]  = sigma_preds_sq[mask]
        sigma_target_sq[mask] = 0

        mask = sigma_preds_sq < eps
        g[mask]          = 0
        sigma_v_sq[mask] = 0

        mask = g < 0
        sigma_v_sq[mask] = sigma_preds_sq[mask]
        g[mask]          = 0
        sigma_v_sq       = torch.clamp(sigma_v_sq, min=eps)

        preds_vif_scale = torch.log10(
            1.0 + (g ** 2.0) * sigma_target_sq / (sigma_v_sq + sigma_n_sq)
        )
        preds_vif  = preds_vif  + torch.sum(preds_vif_scale, dim=[1, 2, 3])
        target_vif = target_vif + torch.sum(
            torch.log10(1.0 + sigma_target_sq / sigma_n_sq), dim=[1, 2, 3]
        )
    return preds_vif / target_vif


def visual_information_fidelity(preds: Tensor, target: Tensor, sigma_n_sq: float = 2.0) -> Tensor:
    """Pixel-based Visual Information Fidelity (VIF)."""
    if preds.size(-1) < 41 or preds.size(-2) < 41:
        raise ValueError(
            f"Invalid size of preds. Expected at least 41x41, got {preds.size(-1)}x{preds.size(-2)}!"
        )
    if target.size(-1) < 41 or target.size(-2) < 41:
        raise ValueError(
            f"Invalid size of target. Expected at least 41x41, got {target.size(-1)}x{target.size(-2)}!"
        )
    per_channel = [
        _vif_per_channel(preds[:, i, :, :], target[:, i, :, :], sigma_n_sq)
        for i in range(preds.size(1))
    ]
    return reduce(torch.cat(per_channel), "elementwise_mean")


def VIF(preds, target):
    """Wrapper around visual_information_fidelity (unchanged from notebook)."""
    return visual_information_fidelity(preds, target)
