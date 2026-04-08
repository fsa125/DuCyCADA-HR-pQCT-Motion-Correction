"""
train.py
--------
Main training script for DuCyCADA — Dual Cycle-Consistent Adversarial
Domain Adaptation network for medical image motion-artifact correction.

Architecture overview
---------------------
DuCyCADA adapts a CycleGAN-style framework with an additional domain-alignment
discriminator (D_forward_DA) and KL-divergence loss to close the distribution
gap between source (simulated) and target (real scanner) domains.

Network components:
    G_forward  (G_F): Source LR → clean HR     (X_s → Ŷ_s, X_t → Ŷ_t)
    G_backward (G_B): HR / Target → source LR  (Ŷ_t → X̂_t, Ŷ_s → X̂_s)
    D_forward  (D_H): Real/fake HR discriminator
    D_backward (D_B): Real/fake LR / target discriminator
    D_forward_DA (D_A): Domain alignment discriminator (source vs target HR)

Loss terms:
    Adversarial  : MSE-based LSGAN losses for G_F, G_B, D_H, D_B
    KL domain    : KL-divergence via D_A to align source and target HR distributions
    Cycle        : L2 cycle-consistency (forward and backward)
    Identity     : L2 identity mapping regularization
    TV           : Total-variation smoothness regularization on generated images

Usage
-----
    python train.py                          # use defaults from config.py
    python train.py --resume checkpoint.pth  # resume from a saved checkpoint
"""

import os
import random
import itertools

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from CinCGAN_pytorch.network import ResnetGenerator, Discriminator
from CinCGAN_pytorch.utils import TVLoss
from ducycada.datasets import ImageDataset_DA, ImageDataset
from ducycada import config


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
random.seed(42)
torch.manual_seed(42)


# ---------------------------------------------------------------------------
# Device setup
# ---------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cuda   = torch.cuda.is_available()
print(f"[DuCyCADA] Using device: {device}")


# ---------------------------------------------------------------------------
# Output directories
# ---------------------------------------------------------------------------
os.makedirs("images",       exist_ok=True)
os.makedirs("saved_models", exist_ok=True)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def build_dataloaders(cfg):
    """Build and return train and test DataLoaders.

    Args:
        cfg: Config module with all path and hyper-parameter attributes.

    Returns:
        Tuple (train_dataloader, test_dataloader).
    """
    import glob

    train_paths_hr  = glob.glob(cfg.DATASET_PATH + cfg.TRAIN_HR_GLOB)
    train_paths_lr  = glob.glob(cfg.DATASET_PATH + cfg.TRAIN_LR_GLOB)
    target_paths    = glob.glob(cfg.DATASET_PATH + cfg.TARGET_GLOB)
    test_paths_hr   = glob.glob(cfg.DATASET_PATH + cfg.TEST_HR_GLOB)
    test_paths_lr   = glob.glob(cfg.DATASET_PATH + cfg.TEST_LR_GLOB)

    hr_shape = (cfg.HR_HEIGHT, cfg.HR_WIDTH)

    train_dataloader = DataLoader(
        ImageDataset_DA(train_paths_hr, train_paths_lr, target_paths, hr_shape=hr_shape),
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
    )
    test_dataloader = DataLoader(
        ImageDataset(test_paths_hr, test_paths_lr, hr_shape=hr_shape),
        batch_size=1,
        shuffle=False,
    )
    return train_dataloader, test_dataloader


def preload_dataset_to_gpu(train_dataloader, save_path: str, cfg) -> list:
    """Pre-load the entire training set into GPU tensors and save to disk.

    This avoids repeated disk I/O during the high-iteration training loop.
    Each element is a TensorDataset of (X_s, Y_s, X_t) triplets.

    Args:
        train_dataloader: DataLoader that yields {'lr', 'hr', 'hr_t'} dicts.
        save_path:        Path where the stacked dataset will be saved.
        cfg:              Config module (used for Tensor type).

    Returns:
        List of TensorDataset objects, one per training sample.
    """
    Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

    print("[DuCyCADA] Pre-loading training data to GPU …")
    stacked_all = []
    for imgs in tqdm(train_dataloader, desc="Pre-loading"):
        imgs_lr  = Variable(imgs["lr"].type(Tensor))
        imgs_hr  = Variable(imgs["hr"].type(Tensor))
        imgs_hr_t = Variable(imgs["hr_t"].type(Tensor))
        stacked_all.append(TensorDataset(imgs_lr, imgs_hr, imgs_hr_t))

    torch.save(stacked_all, save_path)
    print(f"[DuCyCADA] Dataset saved to {save_path}")
    return stacked_all


# ---------------------------------------------------------------------------
# Model / optimizer initialization
# ---------------------------------------------------------------------------

def build_models(cfg):
    """Instantiate all DuCyCADA sub-networks and move them to device.

    Returns:
        Dictionary with keys: G_forward, G_backward, D_forward,
        D_backward, D_forward_DA.
    """
    models = {
        "G_forward":    ResnetGenerator(input_nc=1, output_nc=1).to(device),
        "G_backward":   ResnetGenerator(input_nc=1, output_nc=1).to(device),
        "D_forward":    Discriminator(input_nc=1).to(device),
        "D_backward":   Discriminator(input_nc=1).to(device),
        "D_forward_DA": Discriminator(input_nc=1).to(device),
    }
    return models


def build_optimizers(models, cfg):
    """Build joint Adam optimizers for generators and discriminators.

    Generators (G_F and G_B) share one optimizer; all discriminators share
    another. ReduceLROnPlateau schedulers are attached to each.

    Args:
        models: Dict returned by build_models().
        cfg:    Config module.

    Returns:
        Tuple (G_optim, D_optim, G_scheduler, D_scheduler).
    """
    G_optim = torch.optim.Adam(
        itertools.chain(
            models["G_forward"].parameters(),
            models["G_backward"].parameters(),
        ),
        lr=cfg.LR, betas=(cfg.B1, cfg.B2), eps=cfg.EPS,
    )
    D_optim = torch.optim.Adam(
        itertools.chain(
            models["D_forward"].parameters(),
            models["D_backward"].parameters(),
        ),
        lr=cfg.LR, betas=(cfg.B1, cfg.B2), eps=cfg.EPS,
    )

    G_scheduler = ReduceLROnPlateau(G_optim, mode="min", factor=0.5, patience=5, verbose=True)
    D_scheduler = ReduceLROnPlateau(D_optim, mode="min", factor=0.5, patience=5, verbose=True)

    return G_optim, D_optim, G_scheduler, D_scheduler


def build_losses():
    """Instantiate and return all loss criteria used in DuCyCADA training.

    Returns:
        Dict with keys: GAN, content, TV, KL.
    """
    losses = {
        "GAN":     torch.nn.MSELoss().to(device),    # LSGAN adversarial loss
        "content": torch.nn.L1Loss().to(device),     # Pixel-level reconstruction
        "TV":      TVLoss().to(device),              # Total-variation regularization
        "KL":      torch.nn.KLDivLoss(reduction="batchmean").to(device),  # Domain alignment
    }
    return losses


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------

def save_checkpoint(epoch: int, models: dict, G_optim, D_optim, path: str):
    """Save a full training checkpoint (models + optimizers + epoch).

    Args:
        epoch:   Current training epoch (0-indexed).
        models:  Dict of sub-networks.
        G_optim: Generator optimizer.
        D_optim: Discriminator optimizer.
        path:    Destination .pth file path.
    """
    torch.save({
        "epoch":                  epoch,
        "G_forward_state_dict":   models["G_forward"].state_dict(),
        "G_backward_state_dict":  models["G_backward"].state_dict(),
        "D_forward_state_dict":   models["D_forward"].state_dict(),
        "D_backward_state_dict":  models["D_backward"].state_dict(),
        "D_forward_DA_state_dict": models["D_forward_DA"].state_dict(),
        "G_optimizer_state_dict": G_optim.state_dict(),
        "D_optimizer_state_dict": D_optim.state_dict(),
    }, path)
    print(f"[DuCyCADA] Checkpoint saved → {path}")


def load_checkpoint(path: str, models: dict, G_optim, D_optim):
    """Resume training from a saved checkpoint.

    Args:
        path:    Path to the .pth checkpoint file.
        models:  Dict of sub-networks (modified in-place).
        G_optim: Generator optimizer (modified in-place).
        D_optim: Discriminator optimizer (modified in-place).

    Returns:
        The epoch stored in the checkpoint (int).
    """
    checkpoint = torch.load(path)

    models["G_forward"].load_state_dict(checkpoint["G_forward_state_dict"])
    models["G_backward"].load_state_dict(checkpoint["G_backward_state_dict"])
    models["D_forward"].load_state_dict(checkpoint["D_forward_state_dict"])
    models["D_backward"].load_state_dict(checkpoint["D_backward_state_dict"])
    models["D_forward_DA"].load_state_dict(checkpoint["D_forward_DA_state_dict"])

    G_optim.load_state_dict(checkpoint["G_optimizer_state_dict"])
    D_optim.load_state_dict(checkpoint["D_optimizer_state_dict"])

    start_epoch = checkpoint["epoch"]
    print(f"[DuCyCADA] Resumed from epoch {start_epoch}")
    return start_epoch


# ---------------------------------------------------------------------------
# Single training step
# ---------------------------------------------------------------------------

def train_step(
    imgs_lr, imgs_hr, imgs_lr_t,
    models, losses,
    G_optim, D_optim,
    g_scaler: GradScaler, d_scaler: GradScaler,
):
    """Execute one DuCyCADA training iteration (discriminator + generator update).

    Notation used in comments:
        X_s   = source LR (simulated motion-corrupted) image
        Y_s   = source HR (clean) image
        X_t   = target domain LR image (unpaired)
        Ŷ_s   = G_F(X_s)  — predicted HR from source
        Ŷ_t   = G_F(X_t)  — predicted HR from target
        X̂_t   = G_B(Ŷ_t)  — cycle-reconstructed target
        X̂_s   = G_B(Ŷ_s)  — cycle-reconstructed source

    Args:
        imgs_lr:   Source LR batch tensor (X_s).
        imgs_hr:   Source HR batch tensor (Y_s).
        imgs_lr_t: Target LR batch tensor (X_t).
        models:    Dict of sub-networks.
        losses:    Dict of loss criteria.
        G_optim:   Generator optimizer.
        D_optim:   Discriminator optimizer.
        g_scaler:  AMP GradScaler for generators.
        d_scaler:  AMP GradScaler for discriminators.

    Returns:
        Tuple (d_loss_val, g_loss_val) as Python floats.
    """
    G_F  = models["G_forward"]
    G_B  = models["G_backward"]
    D_H  = models["D_forward"]
    D_B  = models["D_backward"]
    D_A  = models["D_forward_DA"]

    crit_GAN = losses["GAN"]
    crit_KL  = losses["KL"]
    TV_Loss  = losses["TV"]

    # ------------------------------------------------------------------ #
    #  1. Discriminator update                                             #
    # ------------------------------------------------------------------ #
    D_optim.zero_grad()

    with autocast():
        # --- Forward passes ---
        fake_fwd   = G_F(imgs_lr)            # Ŷ_s  = G_F(X_s)
        fake_fwd_t = G_F(imgs_lr_t)          # Ŷ_t  = G_F(X_t)
        fake_bwd   = G_B(fake_fwd_t)         # X̂_t  = G_B(Ŷ_t)
        fake_bwd_s = G_B(fake_fwd.detach())  # X̂_s  = G_B(Ŷ_s)

        # --- Discriminator outputs ---
        # D_H: real HR source vs. fake HR (from source)
        real_out     = D_H(imgs_hr)
        fake_out     = D_H(fake_fwd.detach())

        # D_B: real target LR vs. cycle-reconstructed target
        real_out_back   = D_B(imgs_lr_t)
        fake_out_back   = D_B(fake_bwd.detach())

        # D_B: real source LR vs. cycle-reconstructed source
        real_out_back_s = D_B(imgs_lr)
        fake_out_back_s = D_B(fake_bwd_s.detach())

        # D_A (domain alignment): target HR vs. source HR outputs
        real_out_d = D_A(fake_fwd_t.detach())   # "real" in target domain = Ŷ_t
        fake_out_d = D_A(fake_fwd.detach())     # "fake" in target domain = Ŷ_s

        # --- Adversarial losses (LSGAN: real=1, fake=0) ---
        d_loss_real      = crit_GAN(real_out,       torch.ones_like(real_out))
        d_loss_fake      = crit_GAN(fake_out,       torch.zeros_like(fake_out))
        d_loss_back_real = crit_GAN(real_out_back,  torch.ones_like(real_out_back))
        d_loss_back_fake = crit_GAN(fake_out_back,  torch.zeros_like(fake_out_back))
        d_loss_back_real_s = crit_GAN(real_out_back_s, torch.ones_like(real_out_back))
        d_loss_back_fake_s = crit_GAN(fake_out_back_s, torch.zeros_like(fake_out_back))

        # --- KL domain-alignment loss (align D_A outputs to uniform distribution) ---
        d_loss_domain = (
            crit_KL(F.log_softmax(fake_out_d, dim=1),
                    F.softmax(torch.zeros_like(fake_out_d), dim=1)) +
            crit_KL(F.log_softmax(real_out_d, dim=1),
                    F.softmax(torch.ones_like(real_out_d), dim=1))
        )

        # --- Total discriminator loss (weighted sum) ---
        d_loss = (
            (d_loss_real + d_loss_fake)
            + 0.1 * (d_loss_back_real + d_loss_back_fake)
            + 0.1 * d_loss_domain
            + 0.3 * (d_loss_back_real_s + d_loss_back_fake_s)
        )

    d_scaler.scale(d_loss).backward()
    d_scaler.step(D_optim)
    d_scaler.update()

    # ------------------------------------------------------------------ #
    #  2. Generator update                                                 #
    # ------------------------------------------------------------------ #
    G_optim.zero_grad()

    with autocast():
        # Fresh forward pass (no detach — gradients flow to generators)
        fake_fwd   = G_F(imgs_lr)
        fake_fwd_t = G_F(imgs_lr_t)
        fake_bwd   = G_B(fake_fwd_t)
        fake_bwd_s = G_B(fake_fwd)

        # --- Generator adversarial losses (fool discriminators: target=1) ---
        g_loss_adv       = crit_GAN(D_H(fake_fwd),    torch.ones_like(D_H(fake_fwd)))
        g_loss_adv_back  = crit_GAN(D_B(fake_bwd),    torch.ones_like(D_B(fake_bwd)))
        g_loss_adv_back_s = crit_GAN(D_B(fake_bwd_s), torch.ones_like(D_B(fake_bwd_s)))

        # --- KL domain alignment for generators ---
        g_output_d = D_A(fake_fwd)
        g_loss_domain = crit_KL(
            F.log_softmax(g_output_d, dim=1),
            F.softmax(torch.ones_like(g_output_d), dim=1)
        )

        # --- Cycle-consistency losses ---
        cycle_fwd = crit_GAN(fake_bwd_s, imgs_lr)          # G_B(G_F(X_s)) ≈ X_s
        cycle_bwd = crit_GAN(G_B(fake_fwd_t), imgs_lr_t)   # G_B(G_F(X_t)) ≈ X_t

        # --- Identity mapping losses ---
        identity_fwd = crit_GAN(G_F(imgs_hr),    imgs_hr)     # G_F(Y_s) ≈ Y_s
        identity_bwd = crit_GAN(G_B(imgs_lr_t),  imgs_lr_t)   # G_B(X_t) ≈ X_t

        # --- TV regularization on generated images ---
        tv_loss   = TV_Loss(fake_fwd)
        tv_loss_t = TV_Loss(fake_fwd_t)

        # --- Total generator loss (weighted combination) ---
        g_loss = (
            g_loss_adv
            + 0.01 * g_loss_adv_back
            + 0.01 * g_loss_domain
            + 0.50 * cycle_bwd
            + 0.40 * cycle_fwd
            + 0.20 * identity_fwd
            + 0.10 * identity_bwd
            + 2.00 * tv_loss
            + 2.00 * tv_loss_t
        )

    g_scaler.scale(g_loss).backward()
    g_scaler.step(G_optim)
    g_scaler.update()

    return d_loss.item(), g_loss.item()


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(cfg, resume_path: str = None):
    """Run the full DuCyCADA training loop.

    Args:
        cfg:         Config module with all hyper-parameters and paths.
        resume_path: Optional path to a checkpoint .pth file to resume from.
    """
    # --- Build components ---
    models                         = build_models(cfg)
    G_optim, D_optim, G_sch, D_sch = build_optimizers(models, cfg)
    losses                         = build_losses()

    g_scaler = GradScaler()
    d_scaler = GradScaler()

    # --- Optionally resume from checkpoint ---
    start_epoch = 0
    if resume_path and os.path.isfile(resume_path):
        start_epoch = load_checkpoint(resume_path, models, G_optim, D_optim)

    # --- Load pre-computed GPU dataset ---
    stacked_dataset = torch.load(cfg.STACKED_DATASET_PATH)
    n_source = cfg.N_SOURCE_SAMPLES     # length of source split
    n_target = cfg.N_TARGET_SAMPLES     # length of target split

    n_epochs    = cfg.N_EPOCHS
    best_g_loss = float("inf")

    print(f"[DuCyCADA] Starting training from epoch {start_epoch} / {n_epochs}")
    for epoch in tqdm(range(start_epoch, n_epochs), desc="Training"):

        # Set all sub-networks to training mode
        for m in models.values():
            m.train()

        # --- Sample one source and one target image at random ---
        idx   = random.randint(0, n_source - 1)
        idx_t = random.randint(0, n_target - 1)

        imgs_lr   = stacked_dataset[idx].tensors[0].to(device)    # X_s
        imgs_hr   = stacked_dataset[idx].tensors[1].to(device)    # Y_s
        imgs_lr_t = stacked_dataset[idx_t].tensors[2].to(device)  # X_t

        # --- One optimization step ---
        d_loss_val, g_loss_val = train_step(
            imgs_lr, imgs_hr, imgs_lr_t,
            models, losses,
            G_optim, D_optim,
            g_scaler, d_scaler,
        )

        # --- Console logging every 100 epochs ---
        if (epoch + 1) % 100 == 0:
            print(
                f"Epoch [{epoch+1}/{n_epochs}]  "
                f"D_loss: {d_loss_val:.4f}  "
                f"G_loss: {g_loss_val:.4f}"
            )

        # --- Checkpoint every SAVE_INTERVAL epochs ---
        if (epoch + 1) % cfg.SAVE_INTERVAL == 0:
            ckpt_path = f"saved_models/DuCyCADA_checkpoint_epoch_{epoch+1}.pth"
            save_checkpoint(epoch, models, G_optim, D_optim, ckpt_path)

        # --- Track best generator model ---
        if g_loss_val < best_g_loss:
            best_g_loss = g_loss_val
            torch.save(models["G_forward"].state_dict(),
                       cfg.BEST_MODEL_PATH)

    print("[DuCyCADA] Training complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train DuCyCADA")
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to a checkpoint .pth file to resume training from."
    )
    args = parser.parse_args()

    train(config, resume_path=args.resume)
