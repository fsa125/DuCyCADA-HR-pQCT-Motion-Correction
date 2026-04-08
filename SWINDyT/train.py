"""
train.py
--------
SWINDyT pipeline.

Trains the supervised motion-correction network SWINDyT (G_forward_pred)
using the pre-adapted stacked dataset produced by preload_data.py.

Usage
-----
python train.py \\
    --stacked_dataset   stacked_dataset_DA_Dist_RAD.pt \\
    --n_epochs          151200 \\
    --lr                0.00008 \\
    --save_every        15120 \\
    --output_name       SWINIR_CYCADA_KL_generator_Dist_rad_final \\
    --num_workers       4
"""

import argparse
import os
import random

import torch
from torch.autograd import Variable
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from models import build_models
from utils  import apply_fixed_gaussian_blur


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def get_args():
    parser = argparse.ArgumentParser(
        description="Train SWINDyT (G_forward_pred) on the pre-adapted stacked dataset."
    )

    # Data
    parser.add_argument("--stacked_dataset", type=str,
                        default="stacked_dataset_DA_Dist_RAD.pt",
                        help="Path to the .pt stacked dataset produced by preload_data.py.")
    parser.add_argument("--dataset_size", type=int, default=15119,
                        help="Upper index bound for random sampling from stacked_dataset "
                             "(0-indexed, inclusive).  Set to len(stacked_dataset)-1.")

    # Image config
    parser.add_argument("--hr_height", type=int, default=256)
    parser.add_argument("--hr_width",  type=int, default=256)

    # Training hyper-parameters
    parser.add_argument("--n_epochs",   type=int,   default=151200,
                        help="Total number of training iterations.")
    parser.add_argument("--batch_size", type=int,   default=1)
    parser.add_argument("--lr",         type=float, default=8e-5,
                        help="Adam learning rate.")
    parser.add_argument("--b1",         type=float, default=0.5,
                        help="Adam beta1.")
    parser.add_argument("--b2",         type=float, default=0.999,
                        help="Adam beta2.")
    parser.add_argument("--eps",        type=float, default=1e-8,
                        help="Adam epsilon.")
    parser.add_argument("--seed",       type=int,   default=42)

    # Loss weights  
    parser.add_argument("--lambda_consistency", type=float, default=0.1)
    parser.add_argument("--lambda_tv",          type=float, default=0.2)
    parser.add_argument("--lambda_vgg",         type=float, default=1e-3)

    # Logging / checkpointing
    parser.add_argument("--log_every",  type=int, default=100,
                        help="Print loss every N iterations.")
    parser.add_argument("--save_every", type=int, default=15120,
                        help="Save model checkpoint every N iterations.")
    parser.add_argument("--output_dir", type=str, default="saved_models",
                        help="Directory to save model checkpoints.")
    parser.add_argument("--output_name", type=str,
                        default="SWINIR_CYCADA_KL_generator_Dist_rad_final",
                        help="Base name for saved checkpoint files.")

    # Pretrained weights (optional)
    parser.add_argument("--pred_weights", type=str, default=None,
                        help="(Optional) Path to pretrained G_forward_pred weights to resume from.")

    # Misc
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader worker processes (used for future dataloader calls).")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = get_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cuda   = torch.cuda.is_available()

    # Build models
    m = build_models(args)
    G_forward_pred    = m["G_forward_pred"]
    D_forward         = m["D_forward"]
    feature_extractor = m["feature_extractor"]
    criterion_GAN     = m["criterion_GAN"]
    criterion_content = m["criterion_content"]
    TV_Loss           = m["TV_Loss"]
    G_optim           = m["G_optim"]
    D_optim           = m["D_optim"]

    # Optionally resume from checkpoint
    if args.pred_weights:
        G_forward_pred.load_state_dict(torch.load(args.pred_weights, map_location=device))
        print(f"Resumed G_forward_pred from: {args.pred_weights}")

    # Load stacked dataset
    print(f"Loading stacked dataset from: {args.stacked_dataset}")
    stacked_dataset = torch.load(args.stacked_dataset)
    print(f"  -> {len(stacked_dataset)} samples loaded.")

    # AMP scalers
    g_scaler = GradScaler()
    d_scaler = GradScaler()

    train_gen_losses, train_disc_losses, train_counter = [], [], []
    CRITIC = 5  # (kept for reference; currently unused in this training variant)

    # -----------------------------------------------------------------------
    # Training loop  (unchanged logic from notebook cell 29)
    # -----------------------------------------------------------------------
    print(f"Starting training for {args.n_epochs} iterations ...")
    for epoch in tqdm(range(args.n_epochs)):
        gen_loss = 0
        disc_loss = 0
        G_forward_pred.train()
        D_forward.train()

        # Random sample
        random_integer = random.randint(0, args.dataset_size)
        imgs_lr = stacked_dataset[random_integer].tensors[0].cuda() if cuda else \
                  stacked_dataset[random_integer].tensors[0]
        imgs_hr = stacked_dataset[random_integer].tensors[1].cuda() if cuda else \
                  stacked_dataset[random_integer].tensors[1]

        # -- Discriminator update --
        D_optim.zero_grad()
        with autocast():
            fake_forward = G_forward_pred(imgs_lr)

            dis1_forward_output_real = D_forward(imgs_hr)
            dis1_forward_output_fake = D_forward(fake_forward.detach())

            dis1_forward_predicted_real = torch.mean(dis1_forward_output_real, dim=[1, 2, 3])
            dis1_forward_predicted_fake = torch.mean(dis1_forward_output_fake, dim=[1, 2, 3])

            dis1_forward_ad_loss = (
                criterion_GAN(dis1_forward_predicted_fake,
                              torch.zeros_like(dis1_forward_predicted_fake).to(device))
                + criterion_GAN(dis1_forward_predicted_real,
                                torch.ones_like(dis1_forward_predicted_real).to(device))
            )
            dis1_loss = dis1_forward_ad_loss

        d_scaler.scale(dis1_loss).backward()
        d_scaler.step(D_optim)
        d_scaler.update()

        # -- Generator update --
        G_optim.zero_grad()
        with autocast():
            fake_forward = G_forward_pred(imgs_lr)

            gen_features  = feature_extractor(fake_forward)
            real_features = feature_extractor(imgs_hr)
            loss_vgg_content = criterion_content(gen_features, real_features.detach())

            dis1_forward_output_fake    = D_forward(fake_forward)
            dis1_forward_predicted_fake = torch.mean(dis1_forward_output_fake, dim=[1, 2, 3])

            g1_forward_ad_loss  = criterion_GAN(
                dis1_forward_predicted_fake,
                torch.ones_like(dis1_forward_predicted_fake).to(device),
            )
            consistency_loss   = criterion_content(fake_forward, imgs_hr)
            g1_forward_tv_loss = TV_Loss(fake_forward)

            g1_forward_loss = (
                g1_forward_ad_loss
                + args.lambda_consistency * consistency_loss
                + args.lambda_tv  * g1_forward_tv_loss
                + args.lambda_vgg * loss_vgg_content
            )
            g1_loss = g1_forward_loss

        g_scaler.scale(g1_loss).backward()
        g_scaler.step(G_optim)
        g_scaler.update()

        # -- Record --
        gen_loss  += g1_loss.item()
        disc_loss += dis1_loss.item()
        train_gen_losses.append(g1_loss.item())
        train_disc_losses.append(dis1_loss.item())
        train_counter.append(epoch + 1)

        # -- Logging --
        if (epoch + 1) % args.log_every == 0:
            print(
                f"Iter [{epoch + 1}/{args.n_epochs}]  "
                f"Gen Loss: {g1_loss.item():.4f}  "
                f"Disc Loss: {dis1_loss.item():.4f}"
            )

        # -- Checkpoint --
        if (epoch + 1) % args.save_every == 0:
            ckpt_path = os.path.join(
                args.output_dir,
                f"{args.output_name}_iter{epoch + 1}.pth",
            )
            torch.save(G_forward_pred.state_dict(), ckpt_path)
            print(f"Checkpoint saved: {ckpt_path}")

    # Final save
    final_path = os.path.join(args.output_dir, f"{args.output_name}_FINAL.pth")
    torch.save(G_forward_pred.state_dict(), final_path)
    print(f"Training complete. Final model saved to: {final_path}")


if __name__ == "__main__":
    main()
