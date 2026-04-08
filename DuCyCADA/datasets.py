"""
datasets.py
-----------
PyTorch Dataset classes for DuCyCADA (Dual Cycle-Consistent Adversarial
Domain Adaptation) training and evaluation.

Datasets:
    - ImageDataset       : Paired source LR/HR dataset for baseline training.
    - ImageDataset_test  : Single-domain dataset for inference/testing.
    - ImageDataset_DA    : Paired source + unpaired target dataset for DA training.
"""

import glob
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


# ---------------------------------------------------------------------------
# Helper: build a standard resize + normalize transform for grayscale images
# ---------------------------------------------------------------------------
def _make_transform(hr_height: int, hr_width: int) -> transforms.Compose:
    """Returns a torchvision transform pipeline for 1-channel images.

    Args:
        hr_height: Target image height in pixels.
        hr_width:  Target image width in pixels.

    Returns:
        A Compose transform that resizes (BICUBIC), converts to tensor,
        and normalizes to [-1, 1].
    """
    mean = np.array([0.5])
    std  = np.array([0.5])
    return transforms.Compose([
        transforms.Resize((hr_height, hr_width), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


# ---------------------------------------------------------------------------
# Dataset 1 – Paired Source (LR / HR) for supervised / warm-up training
# ---------------------------------------------------------------------------
class ImageDataset(Dataset):
    """Paired LR/HR image dataset for source domain training.

    Args:
        files_hr: List of file paths for high-resolution source images (Y_s).
        files_lr: List of file paths for low-resolution source images (X_s).
        hr_shape: Tuple (height, width) for the output image resolution.
        device:   Device string ('cuda' or 'cpu') to pre-load tensors onto.
    """

    def __init__(self, files_hr, files_lr, hr_shape, device: str = "cuda"):
        hr_height, hr_width = hr_shape
        self.transform  = _make_transform(hr_height, hr_width)
        self.files_hr   = files_hr
        self.files_lr   = files_lr
        self.device     = device

    def __getitem__(self, index):
        # Load as grayscale (1-channel) images
        img_hr = Image.open(self.files_hr[index % len(self.files_hr)]).convert("L")
        img_lr = Image.open(self.files_lr[index % len(self.files_lr)]).convert("L")

        img_hr = self.transform(img_hr).to(self.device)
        img_lr = self.transform(img_lr).to(self.device)

        return {"lr": img_lr, "hr": img_hr}

    def __len__(self):
        return len(self.files_hr)


# ---------------------------------------------------------------------------
# Dataset 2 – Single-domain dataset for inference / test-time evaluation
# ---------------------------------------------------------------------------
class ImageDataset_test(Dataset):
    """Single-domain dataset used during inference.

    Both 'lr' and 'hr' keys return the same image so that standard evaluation
    loops can run unchanged.

    Args:
        files_hr: List of file paths for the images to evaluate.
        hr_shape: Tuple (height, width) for the output image resolution.
        device:   Device string ('cuda' or 'cpu').
    """

    def __init__(self, files_hr, hr_shape, device: str = "cuda"):
        hr_height, hr_width = hr_shape
        self.transform = _make_transform(hr_height, hr_width)
        self.files_hr  = files_hr
        self.device    = device

    def __getitem__(self, index):
        img = Image.open(self.files_hr[index % len(self.files_hr)]).convert("L")
        img = self.transform(img).to(self.device)
        # Return duplicate so evaluation code can use both keys
        return {"lr": img, "hr": img}

    def __len__(self):
        return len(self.files_hr)


# ---------------------------------------------------------------------------
# Dataset 3 – Domain Adaptation dataset (source pairs + unpaired target)
# ---------------------------------------------------------------------------
class ImageDataset_DA(Dataset):
    """Dataset for DuCyCADA domain-adaptation training.

    Yields triplets:
        - ``lr``   : Simulated (motion-corrupted) source image X_s.
        - ``hr``   : Clean source image Y_s.
        - ``hr_t`` : Unpaired target domain image X_t.

    The source images are indexed modulo their lengths; the target images are
    indexed independently so the dataset length equals the source size.

    Args:
        files_hr:   File paths for HR source images (Y_s).
        files_lr:   File paths for LR source images (X_s).
        files_hr_t: File paths for target domain images (X_t).
        hr_shape:   Tuple (height, width) for the output resolution.
        device:     Device string ('cuda' or 'cpu').
    """

    def __init__(self, files_hr, files_lr, files_hr_t, hr_shape, device: str = "cuda"):
        hr_height, hr_width = hr_shape
        self.transform      = _make_transform(hr_height, hr_width)
        self.files_hr       = files_hr
        self.files_lr       = files_lr
        self.files_hr_t     = files_hr_t
        self.device         = device

    def __getitem__(self, index):
        img_hr  = Image.open(self.files_hr[index % len(self.files_hr)]).convert("L")
        img_lr  = Image.open(self.files_lr[index % len(self.files_lr)]).convert("L")
        img_hr_t = Image.open(self.files_hr_t[index % len(self.files_hr_t)]).convert("L")

        img_hr   = self.transform(img_hr).to(self.device)
        img_lr   = self.transform(img_lr).to(self.device)
        img_hr_t = self.transform(img_hr_t).to(self.device)

        return {"lr": img_lr, "hr": img_hr, "hr_t": img_hr_t}

    def __len__(self):
        return len(self.files_hr)
