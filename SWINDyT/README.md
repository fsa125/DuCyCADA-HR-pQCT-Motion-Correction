# DuCyCADA + SWINDyT

Domain-adaptation-based motion correction for high-resolution peripheral quantitative CT (HR-pQCT).

## Overview

This repository contains the training and inference code for the two-stage pipeline described in the paper:

1. **DuCyCADA** — A CycleGAN-based unsupervised domain-adaptation model that bridges the gap between source (simulated) and target (real-world motion-corrupted) domains.
2. **SWINDyT** — A SwinIR-based supervised motion-correction network trained on source-domain data adapted by DuCyCADA.

At inference time, only the trained SWINDyT model is needed.

---

## Repository structure

```
DuCyCADA_SWINDyT/
├── datasets.py        # Dataset classes and dataloader factory
├── models.py          # Network definitions and model builder
├── utils.py           # Image processing and metric utilities (Gaussian blur, VIF, etc.)
├── preload_data.py    # Stage 1: run source images through G_forward_da and cache to disk
├── train.py           # Stage 2: train SWINDyT on the cached adapted dataset
├── test.py            # Stage 3: run inference and save PNG outputs
├── evaluate.py        # Compute PSNR / SSIM / VIF / BRISQUE / NIQE / CLIPQA / LPIPS
├── plot_results.py    # Generate box-plot figures from metric CSVs
├── requirements.txt
└── README.md
```

External dependencies expected alongside this repo:

```
CinCGAN_pytorch/   # ResnetGenerator, UNetDiscriminatorSN, TVLoss
SWINIR/            # network_swinir_1.SwinIR
RFIQ/              # NIQE.BRISQUE, NIQE.NIQE, NIQE.CLIPIQA, NIQE.MUSIQ, NIQE.LPIPS_Simple
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Dataset layout

Place your data under a root directory (e.g. `archive1/`):

```
archive1/
├── Dist rad train img_new/          # Source HR images  (Y_s)
├── Dist rad train img_new simz/     # Source LR images  (X_s, simulated motion)
├── Dist rad target img/             # Target HR images  (real-world, no ground truth)
└── Distal_Radius_imgs_sr/
    ├── Ground Truth/Volume_2/       # Test ground-truth
    └── Motion Corrupted/Volume_2/   # Test motion-corrupted
```

---

## Usage

### Step 1 — Preload adapted training data

Runs every source training image through the pretrained DuCyCADA generator and caches the result.

```bash
python preload_data.py \
    --dataset_path      archive1/ \
    --train_hr_subdir   "Dist rad train img_new" \
    --train_lr_subdir   "Dist rad train img_new simz" \
    --da_weights        saved_models/DA_Dist_rad_generator_KL_Final_RZ_anika.pth \
    --output_pt         stacked_dataset_DA_Dist_RAD.pt \
    --num_workers       4
```

### Step 2 — Train SWINDyT

```bash
python train.py \
    --stacked_dataset   stacked_dataset_DA_Dist_RAD.pt \
    --n_epochs          151200 \
    --lr                0.00008 \
    --save_every        15120 \
    --output_name       SWINIR_CYCADA_KL_generator_Dist_rad \
    --num_workers       4
```

Key arguments:

| Argument | Default | Description |
|---|---|---|
| `--stacked_dataset` | — | `.pt` file from Step 1 |
| `--n_epochs` | 151200 | Total training iterations |
| `--lr` | 8e-5 | Adam learning rate |
| `--lambda_consistency` | 0.1 | Weight for pixel consistency loss |
| `--lambda_tv` | 0.2 | Weight for total variation loss |
| `--lambda_vgg` | 1e-3 | Weight for VGG perceptual loss |
| `--save_every` | 15120 | Save checkpoint every N iterations |
| `--num_workers` | 4 | DataLoader workers |

### Step 3 — Inference

```bash
python test.py \
    --dataset_path   archive1/ \
    --test_hr_subdir "Distal_Radius_imgs_sr/Ground Truth/Volume_2" \
    --test_lr_subdir "Distal_Radius_imgs_sr/Motion Corrupted/Volume_2" \
    --pred_weights   saved_models/SWINIR_CYCADA_KL_generator_Dist_rad_FINAL.pth \
    --output_dir     results/Volume_2 \
    --save_figures \
    --num_workers    4
```

### Step 4 — Evaluate

```bash
python evaluate.py \
    --dataset_path   archive1/ \
    --test_hr_subdir "Distal_Radius_imgs_sr/Ground Truth/Volume_2" \
    --test_lr_subdir "Distal_Radius_imgs_sr/Motion Corrupted/Volume_2" \
    --pred_weights   saved_models/SWINIR_CYCADA_KL_generator_Dist_rad_FINAL.pth \
    --output_csv     results/metrics_Dist_rad.csv \
    --num_workers    4
```

### Step 5 — Plot results

```bash
# Single CSV
python plot_results.py \
    --mode   single \
    --csv    results/metrics_Dist_rad.csv \
    --output results/boxplot_Dist_rad.png

# Comparison across anatomical sites
python plot_results.py \
    --mode          compare \
    --csv           results/Metric.csv \
    --cols_a        PSNR_proxrad SSIM_proxrad IFC_proxrad \
    --cols_b        PSNR_distrad SSIM_distrad IFC_distrad \
    --metric_labels PSNR SSIM IFC \
    --output        results/boxplot_compare.png
```

---

## Dataset — HR-MoCo47K

| Region | Subjects | Images | VGS |
|---|---|---|---|
| Distal Radius (S) | 103 | 17,304 | 1 |
| Distal Tibia  (S) | 126 | 21,168 | 1 |
| Distal Radius (T) |  40 |  6,720 | ≥2 |
| Distal Tibia  (T) |  14 |  2,352 | ≥2 |
| **Total**         | **283** | **47,544** | — |

---

## Citation

If you use this code, please cite the corresponding paper.
