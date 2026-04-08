"""
test.py
-------
Stage 3 of the DuCyCADA / SWINDyT pipeline.

Runs inference with the trained G_forward_pred on the test split, saves
per-image PNG outputs, and optionally plots a 4-panel comparison figure
(corrupted | predicted | ground truth | difference).

Usage
-----
python test.py \\
    --dataset_path   archive1/ \\
    --test_hr_subdir "Distal_Radius_imgs_sr/Ground Truth/Volume_2" \\
    --test_lr_subdir "Distal_Radius_imgs_sr/Motion Corrupted/Volume_2" \\
    --pred_weights   saved_models/SWINIR_CYCADA_KL_generator_Dist_rad_final_FINAL.pth \\
    --output_dir     results/Volume_2 \\
    --save_figures   \\
    --num_workers    4
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image

from datasets import ImageDataset
from models   import build_models
from utils    import (apply_fixed_gaussian_blur, normalize_tensor,
                      histogram_matching, show_difference)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def get_args():
    parser = argparse.ArgumentParser(
        description="Run inference with SWINDyT and save results."
    )

    # Dataset
    parser.add_argument("--dataset_path",   type=str, default="archive1/",
                        help="Root directory of the dataset.")
    parser.add_argument("--test_hr_subdir", type=str,
                        default="Distal_Radius_imgs_sr/Ground Truth/Volume_2",
                        help="Sub-folder under dataset_path for test ground-truth HR images.")
    parser.add_argument("--test_lr_subdir", type=str,
                        default="Distal_Radius_imgs_sr/Motion Corrupted/Volume_2",
                        help="Sub-folder under dataset_path for test motion-corrupted LR images.")

    # Image config
    parser.add_argument("--hr_height",  type=int, default=256)
    parser.add_argument("--hr_width",   type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers",type=int, default=4,
                        help="Number of DataLoader worker processes.")

    # Model weights
    parser.add_argument("--pred_weights", type=str,
                        default="saved_models/SWINIR_CYCADA_KL_generator_Dist_rad_final_FINAL.pth",
                        help="Path to trained G_forward_pred weights.")

    # Hyper-parameters (passed to build_models)
    parser.add_argument("--lr",  type=float, default=8e-5)
    parser.add_argument("--b1",  type=float, default=0.5)
    parser.add_argument("--b2",  type=float, default=0.999)
    parser.add_argument("--eps", type=float, default=1e-8)

    # Gaussian blur applied to input at inference (unchanged from notebook)
    parser.add_argument("--blur_sigma",       type=float, default=0.3)
    parser.add_argument("--blur_kernel_size", type=int,   default=5)

    # Output
    parser.add_argument("--output_dir",   type=str,  default="results/Volume_Pred",
                        help="Directory to save prediction PNG images.")
    parser.add_argument("--save_figures", action="store_true",
                        help="If set, also save 4-panel comparison figures.")
    parser.add_argument("--fig_dir",      type=str, default="results/Figures",
                        help="Directory for 4-panel comparison figures (requires --save_figures).")
    parser.add_argument("--fig_dpi",      type=int, default=300)

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args   = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.output_dir, exist_ok=True)
    if args.save_figures:
        os.makedirs(args.fig_dir, exist_ok=True)

    # Build model and load weights
    m              = build_models(args)
    G_forward_pred = m["G_forward_pred"]
    G_forward_pred.load_state_dict(torch.load(args.pred_weights, map_location=device))
    G_forward_pred.eval()
    print(f"Loaded G_forward_pred from: {args.pred_weights}")

    # Dataloader
    hr_shape = (args.hr_height, args.hr_width)
    import glob
    test_paths_hr = glob.glob(args.dataset_path + args.test_hr_subdir + "/*.*")
    test_paths_lr = glob.glob(args.dataset_path + args.test_lr_subdir + "/*.*")

    test_dataloader = DataLoader(
        ImageDataset(test_paths_hr, test_paths_lr, hr_shape=hr_shape,
                     device=str(device)),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    print(f"Test set: {len(test_dataloader)} batches.")

    # -----------------------------------------------------------------------
    # Inference loop  (unchanged logic from notebook cells 68 / 70)
    # -----------------------------------------------------------------------
    epoch    = 0  # kept for tqdm label consistency
    tqdm_bar = tqdm(test_dataloader, desc=f"Testing", total=len(test_dataloader))

    for batch_idx, imgs in enumerate(tqdm_bar):
        with torch.no_grad():
            imgs_lr = Variable(imgs["lr"].to(device))
            imgs_hr = Variable(imgs["hr"].to(device))

            gen_hr = G_forward_pred(
                apply_fixed_gaussian_blur(imgs_lr,
                                          kernel_size=args.blur_kernel_size,
                                          sigma=args.blur_sigma)
            )

            gen_hr     = normalize_tensor(gen_hr)
            t_imgs_hr  = normalize_tensor(imgs_hr)
            imgs_lr_n  = normalize_tensor(imgs_lr)

        # Convert to numpy  (batch dim always 1 during test)
        img_hr_np  = gen_hr.detach().cpu().numpy().transpose(0, 2, 3, 1)
        img_thr_np = t_imgs_hr.detach().cpu().numpy().transpose(0, 2, 3, 1)
        img_tlr_np = imgs_lr_n.detach().cpu().numpy().transpose(0, 2, 3, 1)

        matched = histogram_matching(img_thr_np[0], img_hr_np[0])

        # Save prediction PNG
        out = matched.astype(np.float32)
        out = (out - out.min()) / (out.max() - out.min() + 1e-8)
        out = (out * 255).round().astype(np.uint8)
        Image.fromarray(out.squeeze(-1)).save(
            os.path.join(args.output_dir, f"SWINDyT_{batch_idx:05d}.png")
        )

        # Optional 4-panel comparison figure
        if args.save_figures:
            fig, axes = plt.subplots(1, 4, figsize=(15, 5), dpi=args.fig_dpi)

            axes[0].imshow(img_tlr_np[0], cmap="gray")
            axes[0].set_title("Testing data"); axes[0].axis("off")

            axes[1].imshow(matched, cmap="gray")
            axes[1].set_title("Predicted data"); axes[1].axis("off")

            axes[2].imshow(img_thr_np[0], cmap="gray")
            axes[2].set_title("True image"); axes[2].axis("off")

            diff_img = show_difference(
                img_thr_np[0],
                histogram_matching(img_thr_np[0], img_hr_np[0]),
            )
            im = axes[3].imshow(diff_img, cmap="cividis")
            axes[3].set_title("Difference image"); axes[3].axis("off")

            cbar = fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)
            cbar.set_label("Difference intensity")

            fig.savefig(
                os.path.join(args.fig_dir, f"compare_{batch_idx:05d}.png"),
                dpi=args.fig_dpi,
                bbox_inches="tight",
            )
            plt.close(fig)

    print(f"Done. Predictions saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
