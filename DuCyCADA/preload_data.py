"""
preload_data.py
---------------
Pre-loads the entire DuCyCADA training dataset into GPU tensors and
serializes the result to disk as a list of TensorDatasets.

This one-time step avoids repeated disk I/O during training, which is
critical when running millions of single-sample iterations. The resulting
.pt file is loaded at the start of train.py via ``torch.load()``.

Memory note
-----------
Each TensorDataset stores three (1, 256, 256) float32 tensors on the GPU.
For ~15k source images the GPU footprint is roughly:
    15120 × 3 × (256 × 256 × 4 bytes) ≈ 11.8 GB
Adjust DATASET_PATH / glob patterns in config.py if your GPU has less memory,
or set ``device='cpu'`` in ImageDataset_DA to keep tensors in RAM instead.

Usage
-----
    python preload_data.py
"""

import glob
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from ducycada.datasets import ImageDataset_DA
from ducycada import config


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cuda   = torch.cuda.is_available()
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    hr_shape = (config.HR_HEIGHT, config.HR_WIDTH)

    # ------------------------------------------------------------------ #
    #  Collect file paths                                                  #
    # ------------------------------------------------------------------ #
    train_paths_hr = glob.glob(config.DATASET_PATH + config.TRAIN_HR_GLOB)
    train_paths_lr = glob.glob(config.DATASET_PATH + config.TRAIN_LR_GLOB)
    target_paths   = glob.glob(config.DATASET_PATH + config.TARGET_GLOB)

    print(f"[preload] Source HR images  : {len(train_paths_hr)}")
    print(f"[preload] Source LR images  : {len(train_paths_lr)}")
    print(f"[preload] Target images     : {len(target_paths)}")

    # ------------------------------------------------------------------ #
    #  DataLoader (batch_size=1 to preserve per-image tensors)            #
    # ------------------------------------------------------------------ #
    train_dataloader = DataLoader(
        ImageDataset_DA(
            train_paths_hr, train_paths_lr, target_paths, hr_shape=hr_shape
        ),
        batch_size=1,
        shuffle=True,
    )

    # ------------------------------------------------------------------ #
    #  Iterate and accumulate GPU tensors                                  #
    # ------------------------------------------------------------------ #
    print("[preload] Pre-loading training data to GPU …")
    stacked_all = []
    for imgs in tqdm(train_dataloader, desc="Pre-loading"):
        imgs_lr   = Variable(imgs["lr"].type(Tensor))
        imgs_hr   = Variable(imgs["hr"].type(Tensor))
        imgs_hr_t = Variable(imgs["hr_t"].type(Tensor))
        stacked_all.append(TensorDataset(imgs_lr, imgs_hr, imgs_hr_t))

    # ------------------------------------------------------------------ #
    #  Save to disk                                                        #
    # ------------------------------------------------------------------ #
    save_path = config.STACKED_DATASET_PATH
    torch.save(stacked_all, save_path)
    print(f"[preload] Saved {len(stacked_all)} TensorDatasets → {save_path}")

    # Verify GPU memory usage
    if cuda:
        alloc_mb = torch.cuda.memory_allocated() / 1024 ** 2
        reserv_mb = torch.cuda.memory_reserved()  / 1024 ** 2
        print(f"[preload] GPU allocated : {alloc_mb:.1f} MB")
        print(f"[preload] GPU reserved  : {reserv_mb:.1f} MB")


if __name__ == "__main__":
    main()
