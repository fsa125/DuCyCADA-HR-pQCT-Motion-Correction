"""
datasets.py
-----------
Dataset classes and dataloader factory for DuCyCADA / SWINDyT.

Three dataset variants are provided:
    - ImageDataset        : paired HR/LR, used for supervised training.
    - ImageDataset_test   : single-path dataset that uses the same file as both HR and LR.
    - ImageDataset_DA     : paired HR/LR source + unpaired HR target, used for domain adaptation.
"""

import glob
import numpy as np
from PIL import Image

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


# ---------------------------------------------------------------------------
# Dataset classes  
# ---------------------------------------------------------------------------

class ImageDataset(Dataset):
    """Paired source HR / LR dataset."""

    def __init__(self, files_hr, files_lr, hr_shape, device="cuda"):
        hr_height, hr_width = hr_shape
        mean = np.array([0.5])
        std  = np.array([0.5])
        self.hr_transform = transforms.Compose([
            transforms.Resize((hr_height, hr_height), Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        self.files_hr = files_hr
        self.files_lr = files_lr
        self.device   = device

    def __getitem__(self, index):
        img_hr = Image.open(self.files_hr[index % len(self.files_hr)]).convert("L")
        img_lr = Image.open(self.files_lr[index % len(self.files_lr)]).convert("L")
        img_lr = self.hr_transform(img_lr).to(self.device)
        img_hr = self.hr_transform(img_hr).to(self.device)
        return {"lr": img_lr, "hr": img_hr}

    def __len__(self):
        return len(self.files_hr)


class ImageDataset_test(Dataset):
    """Single-path test dataset (HR path used for both HR and LR slots)."""

    def __init__(self, files_hr, hr_shape, device="cuda"):
        hr_height, hr_width = hr_shape
        mean = np.array([0.5])
        std  = np.array([0.5])
        self.hr_transform = transforms.Compose([
            transforms.Resize((hr_height, hr_height), Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        self.files_hr = files_hr
        self.files_lr = files_hr
        self.device   = device

    def __getitem__(self, index):
        img_hr = Image.open(self.files_hr[index % len(self.files_hr)]).convert("L")
        img_lr = Image.open(self.files_lr[index % len(self.files_lr)]).convert("L")
        img_lr = self.hr_transform(img_lr).to(self.device)
        img_hr = self.hr_transform(img_hr).to(self.device)
        return {"lr": img_lr, "hr": img_hr}

    def __len__(self):
        return len(self.files_hr)


class ImageDataset_DA(Dataset):
    """Paired source HR/LR + unpaired target HR dataset for domain adaptation."""

    def __init__(self, files_hr, files_lr, files_hr_t, hr_shape, device="cuda"):
        hr_height, hr_width = hr_shape
        mean = np.array([0.5])
        std  = np.array([0.5])
        self.hr_transform = transforms.Compose([
            transforms.Resize((hr_height, hr_height), Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        self.files_hr     = files_hr
        self.files_lr     = files_lr
        self.files_hr_target = files_hr_t
        self.device       = device

    def __getitem__(self, index):
        img_hr = Image.open(self.files_hr[index % len(self.files_hr)]).convert("L")
        img_lr = Image.open(self.files_lr[index % len(self.files_lr)]).convert("L")
        img_lr = self.hr_transform(img_lr).to(self.device)
        img_hr = self.hr_transform(img_hr).to(self.device)

        img_hr_t = Image.open(self.files_hr_target[index % len(self.files_hr_target)]).convert("L")
        img_hr_t = self.hr_transform(img_hr_t).to(self.device)

        return {"lr": img_lr, "hr": img_hr, "hr_t": img_hr_t}

    def __len__(self):
        return len(self.files_hr)


# ---------------------------------------------------------------------------
# Dataloader factory
# ---------------------------------------------------------------------------

def build_dataloaders(args):
    """
    Build train and test DataLoaders from CLI arguments.

    Returns
    -------
    train_dataloader, test_dataloader
    """
    hr_shape = (args.hr_height, args.hr_width)
    device   = "cuda" if args.cuda else "cpu"

    train_paths_hr  = glob.glob(args.dataset_path + args.train_hr_subdir  + "/*.*")
    train_paths_lr  = glob.glob(args.dataset_path + args.train_lr_subdir  + "/*.*")
    target_paths_hr = glob.glob(args.dataset_path + args.target_hr_subdir + "/*.*")
    test_paths_hr   = glob.glob(args.dataset_path + args.test_hr_subdir   + "/*.*")
    test_paths_lr   = glob.glob(args.dataset_path + args.test_lr_subdir   + "/*.*")

    train_dataloader = DataLoader(
        ImageDataset_DA(train_paths_hr, train_paths_lr, target_paths_hr,
                        hr_shape=hr_shape, device=device),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    test_dataloader = DataLoader(
        ImageDataset(test_paths_hr, test_paths_lr, hr_shape=hr_shape, device=device),
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
    )

    return train_dataloader, test_dataloader
