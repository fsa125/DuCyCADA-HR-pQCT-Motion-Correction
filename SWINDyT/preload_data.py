"""
preload_data.py
---------------
Stage 1 of the DuCyCADA pipeline.

Runs every training image through the (already-trained) domain-adaptation
generator G_forward_da and saves the adapted tensors as a list of
TensorDatasets on disk.  The resulting .pt file is consumed by train.py.

Usage
-----
python preload_data.py \\
    --dataset_path      archive1/ \\
    --train_hr_subdir   "Dist rad train img_new" \\
    --train_lr_subdir   "Dist rad train img_new simz" \\
    --target_hr_subdir  "Dist rad target img" \\
    --da_weights        saved_models/DA_Dist_rad_generator_KL_Final_RZ_anika.pth \\
    --output_pt         stacked_dataset_DA_Dist_RAD.pt \\
    --num_workers       4
"""

import argparse
import os

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from datasets import ImageDataset
from models   import ResnetGenerator
from utils    import apply_fixed_gaussian_blur


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def get_args():
    parser = argparse.ArgumentParser(
        description="Preload training data through G_forward_da and save to disk."
    )

    # Dataset
    parser.add_argument("--dataset_path",     type=str, default="archive1/",
                        help="Root directory of the dataset.")
    parser.add_argument("--train_hr_subdir",  type=str, default="Dist rad train img_new",
                        help="Sub-folder under dataset_path for source HR images (Y_s).")
    parser.add_argument("--train_lr_subdir",  type=str, default="Dist rad train img_new simz",
                        help="Sub-folder under dataset_path for source LR images (X_s).")
    parser.add_argument("--target_hr_subdir", type=str, default="Dist rad target img",
                        help="Sub-folder under dataset_path for target HR images.")

    # Image config
    parser.add_argument("--hr_height", type=int, default=256, help="HR image height.")
    parser.add_argument("--hr_width",  type=int, default=256, help="HR image width.")
    parser.add_argument("--batch_size",type=int, default=1,   help="Dataloader batch size.")
    parser.add_argument("--num_workers",type=int,default=4,
                        help="Number of dataloader worker processes.")

    # Model
    parser.add_argument("--da_weights", type=str,
                        default="saved_models/DA_Dist_rad_generator_KL_Final_RZ_anika.pth",
                        help="Path to pretrained G_forward_da weights.")

    # Gaussian blur (applied after DA forward pass)
    parser.add_argument("--blur_sigma",      type=float, default=0.3,
                        help="Sigma for the fixed Gaussian blur applied after the DA pass.")
    parser.add_argument("--blur_kernel_size",type=int,   default=5,
                        help="Kernel size for the fixed Gaussian blur.")

    # Output
    parser.add_argument("--output_pt", type=str,
                        default="stacked_dataset_DA_Dist_RAD.pt",
                        help="Path to save the output .pt stacked dataset.")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args   = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cuda   = torch.cuda.is_available()

    hr_shape = (args.hr_height, args.hr_width)

    # Build train dataloader (source pairs only for preloading)
    import glob
    train_paths_hr = glob.glob(args.dataset_path + args.train_hr_subdir + "/*.*")
    train_paths_lr = glob.glob(args.dataset_path + args.train_lr_subdir + "/*.*")

    train_dataloader = DataLoader(
        ImageDataset(train_paths_hr, train_paths_lr, hr_shape=hr_shape,
                     device="cuda" if cuda else "cpu"),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    # Load DA generator
    G_forward_da = ResnetGenerator(input_nc=1, output_nc=1).to(device)
    G_forward_da.load_state_dict(torch.load(args.da_weights, map_location=device))
    G_forward_da.eval()
    print(f"Loaded G_forward_da weights from: {args.da_weights}")

    # Preload loop  
    print("Pre-loading the data to GPU ...")
    stacked_all = []

    for imgs in tqdm(train_dataloader):
        imgs_lr = Variable(imgs["lr"].float().cuda() if cuda else imgs["lr"].float())
        imgs_lr = G_forward_da(imgs_lr)
        imgs_lr = apply_fixed_gaussian_blur(imgs_lr,
                                            kernel_size=args.blur_kernel_size,
                                            sigma=args.blur_sigma)
        imgs_hr = Variable(imgs["hr"].float().cuda() if cuda else imgs["hr"].float())

        imgs_lr_cpu = imgs_lr.detach().cpu()
        imgs_hr_cpu = imgs_hr.detach().cpu()

        stacked_dataset = TensorDataset(imgs_lr_cpu, imgs_hr_cpu)
        stacked_all.append(stacked_dataset)

    torch.save(stacked_all, args.output_pt)
    print(f"Saved stacked dataset ({len(stacked_all)} samples) to: {args.output_pt}")


if __name__ == "__main__":
    main()
