"""
evaluate.py
-----------
Evaluation script for DuCyCADA — runs the trained G_backward generator on
the test set and computes image quality metrics (PSNR, SSIM, VIF).

For each test image the script produces a 4-panel figure:
    [0] Simulated motion-corrupted input  (X_s or X_t)
    [1] DuCyCADA predicted output         (histogram-matched)
    [2] Ground-truth clean image          (Y reference)
    [3] Absolute pixel difference map     (plasma colormap)

Metrics are computed after histogram matching predicted → ground-truth and
written to a CSV file.

Usage
-----
    python evaluate.py --model saved_models/DA_dist_tib_generator.pth
    python evaluate.py --model saved_models/DA_dist_tib_generator.pth \
                       --output Results/my_run/
"""

import os
import csv
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend for server use
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from CinCGAN_pytorch.network import ResnetGenerator
from ducycada.datasets import ImageDataset
from ducycada.metrics import (
    normalize_tensor, histogram_matching, show_difference, VIF
)
from ducycada import config


# ---------------------------------------------------------------------------
# Matplotlib global style (Times New Roman, 12 pt, as used in the paper)
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif":  "Times New Roman",
    "font.size":   12,
})

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def evaluate(model_path: str, output_dir: str, cfg):
    """Run DuCyCADA evaluation on the test split.

    For every test image the function:
        1. Generates the output with G_backward (the SR / motion-correction path).
        2. Histogram-matches the prediction to the ground-truth.
        3. Computes PSNR, SSIM, and VIF.
        4. Saves a 4-panel figure and accumulates metrics.
    At the end it writes all metrics to ``<output_dir>/metrics.csv``.

    Args:
        model_path: Path to the saved G_backward weights (.pth).
        output_dir: Directory where output figures and the CSV are written.
        cfg:        Config module.
    """
    import glob

    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    #  Model setup                                                         #
    # ------------------------------------------------------------------ #
    G_backward = ResnetGenerator(input_nc=1, output_nc=1).to(device)
    G_backward.load_state_dict(torch.load(model_path, map_location=device))
    G_backward.eval()
    print(f"[DuCyCADA] Loaded model from {model_path}")

    # ------------------------------------------------------------------ #
    #  Data setup                                                          #
    # ------------------------------------------------------------------ #
    hr_shape     = (cfg.HR_HEIGHT, cfg.HR_WIDTH)
    test_paths_lr = glob.glob(cfg.DATASET_PATH + cfg.TEST_LR_GLOB)
    test_paths_hr = glob.glob(cfg.DATASET_PATH + cfg.TEST_HR_GLOB)

    test_dataloader = DataLoader(
        ImageDataset(test_paths_hr, test_paths_lr, hr_shape=hr_shape),
        batch_size=1,
        shuffle=False,
    )
    print(f"[DuCyCADA] Test set size: {len(test_dataloader)} images")

    # ------------------------------------------------------------------ #
    #  Evaluation                                                          #
    # ------------------------------------------------------------------ #
    results = []  # Accumulates (batch_idx, PSNR, SSIM, VIF) rows

    tqdm_bar = tqdm(test_dataloader, desc="Evaluating", total=len(test_dataloader))
    for batch_idx, imgs in enumerate(tqdm_bar):

        # --- Inference ---
        imgs_lr = Variable(imgs["lr"])   # simulated / corrupted input
        imgs_hr = Variable(imgs["hr"])   # ground-truth reference

        with torch.no_grad():
            gen_hr = G_backward(imgs_lr)

        # --- Normalize to [0, 1] for display and metric computation ---
        gen_hr   = normalize_tensor(gen_hr)
        imgs_hr  = normalize_tensor(imgs_hr)
        imgs_lr  = normalize_tensor(imgs_lr)

        # --- Convert to numpy (H, W, C) arrays ---
        def _to_numpy(t):
            return np.transpose(t.detach().cpu().numpy(), (0, 2, 3, 1))

        img_lr_np  = _to_numpy(imgs_lr)[0]   # (H, W, 1)  — corrupted input
        img_hr_np  = _to_numpy(gen_hr)[0]    # (H, W, 1)  — prediction
        img_thr_np = _to_numpy(imgs_hr)[0]   # (H, W, 1)  — ground truth

        # Histogram-match prediction → ground truth for fair comparison
        img_hr_matched = histogram_matching(img_thr_np, img_hr_np)

        # --- 4-panel figure ---
        fig, axes = plt.subplots(1, 4, figsize=(15, 5), dpi=300)

        axes[0].imshow(img_lr_np,       cmap="gray")
        axes[0].set_title("Simulated Motion\nCorrupted Image")
        axes[0].axis("off")

        axes[1].imshow(img_hr_matched,  cmap="gray")
        axes[1].set_title("Predicted Motion\nCompensated Image")
        axes[1].axis("off")

        axes[2].imshow(img_thr_np,      cmap="gray")
        axes[2].set_title("Ground Truth Image")
        axes[2].axis("off")

        diff = show_difference(
            img_lr_np.astype(np.float32),
            img_hr_np.astype(np.float32),
        )
        img_plot = axes[3].imshow(diff, cmap="plasma")
        axes[3].set_title("Difference Image")
        axes[3].axis("off")

        cbar = fig.colorbar(img_plot, ax=axes[3], fraction=0.046, pad=0.04)
        cbar.set_label("Difference intensity")

        fig_path = os.path.join(output_dir, f"DuCyCADA_{batch_idx:04d}.png")
        fig.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        # --- Metrics (2-D single-channel slices) ---
        ref = img_thr_np[:, :, 0]
        pred = histogram_matching(img_thr_np[:, :, 0], img_hr_np[:, :, 0])
        data_range = ref.max() - ref.min()

        psnr_val = peak_signal_noise_ratio(ref, pred, data_range=data_range)

        ssim_val = structural_similarity(
            ref, pred,
            multichannel=False,
            gaussian_weights=True,
            sigma=1.5,
            use_sample_covariance=False,
            data_range=data_range,
        )

        # VIF expects (N, C, H, W) float64 tensors
        ref_t  = torch.from_numpy(
            np.transpose(img_thr_np, (2, 0, 1))
        ).unsqueeze(0).to(torch.float64)
        pred_t = torch.from_numpy(
            np.transpose(img_hr_matched, (2, 0, 1))
        ).unsqueeze(0).to(torch.float64)
        vif_val = VIF(pred_t, ref_t).item()

        results.append([batch_idx, psnr_val, ssim_val, vif_val])
        tqdm_bar.set_postfix(PSNR=f"{psnr_val:.2f}", SSIM=f"{ssim_val:.4f}", VIF=f"{vif_val:.4f}")

    # ------------------------------------------------------------------ #
    #  Save metrics to CSV                                                 #
    # ------------------------------------------------------------------ #
    csv_path = os.path.join(output_dir, "metrics.csv")
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Batch", "PSNR", "SSIM", "VIF"])
        writer.writerows(results)

    # --- Summary statistics ---
    arr = np.array(results)
    print("\n[DuCyCADA] Evaluation summary")
    print(f"  PSNR  mean ± std : {arr[:, 1].mean():.3f} ± {arr[:, 1].std():.3f} dB")
    print(f"  SSIM  mean ± std : {arr[:, 2].mean():.4f} ± {arr[:, 2].std():.4f}")
    print(f"  VIF   mean ± std : {arr[:, 3].mean():.4f} ± {arr[:, 3].std():.4f}")
    print(f"  Results saved to  : {output_dir}")
    print(f"  Metrics CSV       : {csv_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate DuCyCADA")
    parser.add_argument(
        "--model",
        type=str,
        default="saved_models/DA_dist_tib_generator_KL_Final_Z_anika.pth",
        help="Path to the trained G_backward weights (.pth).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=config.RESULTS_DIR,
        help="Output directory for figures and metrics CSV.",
    )
    args = parser.parse_args()

    evaluate(args.model, args.output, config)
