"""
evaluate.py
-----------
Computes image-quality metrics on the test set and writes results to a CSV.

Metrics: PSNR, SSIM, VIF, BRISQUE, NIQE, CLIPQA, LPIPS (Alex).

Usage
-----
python evaluate.py \\
    --dataset_path   archive1/ \\
    --test_hr_subdir "Distal_Radius_imgs_sr/Ground Truth/Volume_2" \\
    --test_lr_subdir "Distal_Radius_imgs_sr/Motion Corrupted/Volume_2" \\
    --pred_weights   saved_models/SWINIR_CYCADA_KL_generator_Dist_rad_final_FINAL.pth \\
    --output_csv     results/metrics_Dist_rad.csv \\
    --num_workers    4
"""

import argparse
import csv
import os

import lpips
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

from RFIQ import NIQE

from datasets import ImageDataset
from models   import build_models
from utils    import (apply_fixed_gaussian_blur, normalize_tensor,
                      histogram_matching, VIF)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def get_args():
    parser = argparse.ArgumentParser(
        description="Evaluate SWINDyT predictions against ground truth."
    )

    # Dataset
    parser.add_argument("--dataset_path",   type=str, default="archive1/")
    parser.add_argument("--test_hr_subdir", type=str,
                        default="Distal_Radius_imgs_sr/Ground Truth/Volume_2",
                        help="Sub-folder for ground-truth HR images.")
    parser.add_argument("--test_lr_subdir", type=str,
                        default="Distal_Radius_imgs_sr/Motion Corrupted/Volume_2",
                        help="Sub-folder for motion-corrupted LR images.")

    # Image config
    parser.add_argument("--hr_height",  type=int, default=256)
    parser.add_argument("--hr_width",   type=int, default=256)
    parser.add_argument("--num_workers",type=int, default=4,
                        help="Number of DataLoader worker processes.")

    # Model
    parser.add_argument("--pred_weights", type=str,
                        default="saved_models/SWINIR_CYCADA_KL_generator_Dist_rad_final_FINAL.pth")

    # Hyper-parameters forwarded to build_models
    parser.add_argument("--lr",  type=float, default=8e-5)
    parser.add_argument("--b1",  type=float, default=0.5)
    parser.add_argument("--b2",  type=float, default=0.999)
    parser.add_argument("--eps", type=float, default=1e-8)

    # Gaussian blur applied to input at inference
    parser.add_argument("--blur_sigma",       type=float, default=0.3)
    parser.add_argument("--blur_kernel_size", type=int,   default=5)

    # Output
    parser.add_argument("--output_csv", type=str,
                        default="results/metrics_Dist_rad.csv",
                        help="Path to write the CSV results file.")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args   = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)

    # Build model
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
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # Initialize metrics on CPU  
    brisque_metric = NIQE.BRISQUE().cpu()
    nique_metric   = NIQE.NIQE().cpu()
    clipqa_metric  = NIQE.CLIPIQA().cpu()
    lpips_fn       = lpips.LPIPS(net="alex")

    results  = []
    tqdm_bar = tqdm(test_dataloader, desc="Evaluating", total=len(test_dataloader))

    for batch_idx, imgs in enumerate(tqdm_bar):
        # Inference  
        imgs_hr = imgs["hr"].to(device)

        with torch.no_grad():
            gen_hr    = G_forward_pred(
                apply_fixed_gaussian_blur(imgs_hr,
                                          kernel_size=args.blur_kernel_size,
                                          sigma=args.blur_sigma)
            )
            gen_hr    = normalize_tensor(gen_hr)
            t_imgs_hr = normalize_tensor(imgs_hr)

        gen_hr_cpu = gen_hr.detach().cpu()
        t_hr_cpu   = t_imgs_hr.detach().cpu()

        gen_hr_np  = gen_hr_cpu[0].permute(1, 2, 0).numpy()
        t_hr_np    = t_hr_cpu[0].permute(1, 2, 0).numpy()
        gen_matched = histogram_matching(t_hr_np, gen_hr_np)

        # PSNR
        psnr_value = peak_signal_noise_ratio(
            t_hr_np[:, :, 0], gen_matched[:, :, 0],
            data_range=t_hr_np[:, :, 0].max() - t_hr_np[:, :, 0].min(),
        )

        # SSIM
        ssim_value = structural_similarity(
            t_hr_np, gen_matched,
            channel_axis=-1,
            gaussian_weights=True,
            sigma=1.5,
            use_sample_covariance=False,
            data_range=t_hr_np.max() - t_hr_np.min(),
            multichannel=True,
        )

        # VIF
        t_tensor = torch.from_numpy(t_hr_np).permute(2, 0, 1).unsqueeze(0).double()
        g_tensor = torch.from_numpy(gen_matched).permute(2, 0, 1).unsqueeze(0).double()
        ifc_value = VIF(g_tensor, t_tensor).item()

        # No-reference metrics (all on CPU)
        brisque_gen  = brisque_metric(gen_hr_cpu)[0].item()
        brisque_true = brisque_metric(t_hr_cpu)[0].item()
        nique_gen    = nique_metric(gen_hr_cpu)[0].item()
        nique_true   = nique_metric(t_hr_cpu)[0].item()
        clipqa_gen   = clipqa_metric(gen_hr_cpu)[0].item()
        clipqa_true  = clipqa_metric(t_hr_cpu)[0].item()
        lpips_value  = lpips_fn(gen_hr_cpu, t_hr_cpu)[0].item()

        results.append([
            batch_idx,
            psnr_value, ssim_value, ifc_value,
            brisque_gen, brisque_true,
            nique_gen,   nique_true,
            clipqa_gen,  clipqa_true,
            lpips_value,
        ])

        del imgs_hr, gen_hr, t_imgs_hr
        torch.cuda.empty_cache()

    # Write CSV
    with open(args.output_csv, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Batch", "PSNR", "SSIM", "VIF",
            "BRISQUE_GEN", "BRISQUE_True",
            "NIQUE_GEN",   "NIQUE_True",
            "CLIPQA_GEN",  "CLIPQA_True",
            "LPIPS_ALEX",
        ])
        writer.writerows(results)

    print(f"Evaluation complete. Results saved to: {args.output_csv}")


if __name__ == "__main__":
    main()
