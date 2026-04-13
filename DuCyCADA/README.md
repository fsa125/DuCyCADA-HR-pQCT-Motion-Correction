# DuCyCADA — Dual Cycle-Consistent Adversarial Domain Adaptation

DuCyCADA is a GAN-based domain adaptation framework for medical image
motion-artifact correction. It adapts a CycleGAN-style cycle-consistency
architecture with an extra domain-alignment discriminator and KL-divergence
loss to close the distribution gap between simulated (source) and real-scanner
(target) domains — without requiring paired target images.

---

## Architecture

```
Source domain         Target domain
   X_s ──────────────────────────── X_t
    │                                │
    ▼         G_forward (G_F)        ▼
   Ŷ_s ◄─── ResnetGenerator ───► Ŷ_t
    │                                │
    ▼         G_backward (G_B)       ▼
   X̂_s ◄─── ResnetGenerator ───► X̂_t
```

| Component       | Role |
|----------------|------|
| `G_forward`    | Source/Target LR → predicted clean HR |
| `G_backward`   | HR / Target LR → reconstructed source LR |
| `D_forward`    | Discriminates real vs. fake HR images |
| `D_backward`   | Discriminates real vs. reconstructed LR images |
| `D_forward_DA` | Domain-alignment discriminator (source HR vs. target HR) |

### Loss terms

| Loss | Weight | Description |
|------|--------|-------------|
| Adversarial (LSGAN) | 1.0 | Real/fake classification for G_F and G_B |
| Cycle-consistency   | 0.4 / 0.5 | Enforce round-trip reconstruction |
| Identity mapping    | 0.2 / 0.1 | Prevent unnecessary style shift |
| KL domain alignment | 0.01 | Align source/target HR distributions via D_A |
| Total variation     | 2.0 | Smooth generated images |

---

## Project structure

```
DuCyCADA/
├── ducycada/
│   ├── __init__.py       — package definition
│   ├── config.py         — all hyper-parameters and paths
│   ├── datasets.py       — PyTorch Dataset classes
│   └── metrics.py        — VIF, PSNR helpers, histogram matching
├── scripts/
│   └── preload_data.py   — one-time GPU pre-loading of training data
├── CinCGAN_pytorch/      — generator / discriminator architectures (external)
│   ├── network.py
│   └── utils.py
├── train.py              — main training script
├── evaluate.py           — evaluation + metric computation
├── requirements.txt
└── README.md
```

---

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure paths

Edit `ducycada/config.py` to point `DATASET_PATH` and the glob patterns to
your data:

```python
DATASET_PATH   = "your/data/root/"
TRAIN_HR_GLOB  = "source_hr/*.*"
TRAIN_LR_GLOB  = "source_lr_simulated/*.*"
TARGET_GLOB    = "target_domain/*.*"
TEST_HR_GLOB   = "test/hr/*.*"
TEST_LR_GLOB   = "test/lr_sim/*.*"
```

### 3. Pre-load dataset to GPU

This step reads all images from disk, converts them to tensors, and saves
the result for fast access during training:

```bash
python scripts/preload_data.py
```

### 4. Train

```bash
python train.py
```

Resume from a checkpoint:

```bash
python train.py --resume saved_models/DuCyCADA_checkpoint_epoch_15000.pth
```

### 5. Evaluate

```bash
python evaluate.py --model saved_models/DA_dist_tib_generator.pth \
                   --output Results/my_run/
```

Outputs:
- 4-panel PNG figures per test image in `--output`
- `metrics.csv` with per-image PSNR, SSIM, and VIF scores
- Summary statistics printed to stdout

---

## Data format

All images are expected as single-channel (grayscale) files (PNG/TIFF/BMP).
The network normalizes them to `[-1, 1]` internally. Folder structure must
match the glob patterns in `config.py`.

---

## Dependencies

See `requirements.txt`. Key packages:

- `torch >= 2.0`
- `torchvision >= 0.15`
- `torchmetrics >= 1.6`
- `scikit-image >= 0.20`
- `opencv-python >= 4.7`

---

## Citation

If you use DuCyCADA in your research, please cite the corresponding paper
(add your citation here).

---

## License


The VIF metric implementation (`ducycada/metrics.py`) is adapted from
[torchmetrics](https://github.com/Lightning-AI/torchmetrics) and
[piq](https://github.com/photosynthesis-team/piq), both under Apache 2.0.
