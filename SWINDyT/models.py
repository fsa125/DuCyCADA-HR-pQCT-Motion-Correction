"""
models.py
---------
Model definitions and factory helpers for DuCyCADA / SWINDyT.

Contains:
    - FeatureExtractor  : VGG-19 based perceptual-feature extractor (unchanged).
    - build_models()    : instantiates G_forward_da, G_forward_pred, D_forward,
                          feature_extractor and all losses / optimizers
"""

import torch
import torch.nn as nn
from torchvision.models import vgg19

from CinCGAN_pytorch.network import ResnetGenerator, UNetDiscriminatorSN
from CinCGAN_pytorch.utils   import TVLoss
from SWINIR.network_swinir_1 import SwinIR


# ---------------------------------------------------------------------------
# Feature extractor  
# ---------------------------------------------------------------------------

class FeatureExtractor(nn.Module):
    """VGG-19 perceptual feature extractor adapted for single-channel input."""

    def __init__(self):
        super(FeatureExtractor, self).__init__()
        vgg19_model  = vgg19(pretrained=True)
        vgg_features = list(vgg19_model.features.children())
        # Adapt first conv to accept 1-channel input
        vgg_features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.feature_extractor = nn.Sequential(*vgg_features[:18])

    def forward(self, img):
        return self.feature_extractor(img)


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_swinir(args):
    """Build the SwinIR prediction network (G_forward_pred)."""
    upscale     = 1
    window_size = 4
    height = (args.hr_height // upscale // window_size + 1) * window_size
    width  = (args.hr_width  // upscale // window_size + 1) * window_size
    model = SwinIR(
        upscale=1,
        img_size=(height, width),
        window_size=window_size,
        img_range=1.0,
        depths=[6, 6, 6, 6],
        embed_dim=60,
        num_heads=[6, 6, 6, 6],
        mlp_ratio=2,
    )
    return model


def build_models(args):
    """
    Instantiate all networks, losses, and optimizers.

    Returns a dict with keys:
        G_forward_da, G_forward_pred, D_forward,
        feature_extractor,
        criterion_GAN, criterion_content, TV_Loss,
        G_optim, D_optim,
        cuda  (bool)
    """
    cuda   = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")

    # Networks
    G_forward_da   = ResnetGenerator(input_nc=1, output_nc=1).to(device)
    D_forward      = UNetDiscriminatorSN(num_in_ch=1).to(device)
    G_forward_pred = build_swinir(args).to(device)
    feature_extractor = FeatureExtractor().to(device)

    # Losses
    criterion_GAN     = torch.nn.MSELoss().to(device)
    criterion_content = torch.nn.L1Loss().to(device)
    TV_Loss_fn        = TVLoss().to(device)

    # Optimizers
    G_optim = torch.optim.Adam(
        G_forward_pred.parameters(),
        lr=args.lr,
        betas=(args.b1, args.b2),
        eps=args.eps,
    )
    D_optim = torch.optim.Adam(
        D_forward.parameters(),
        lr=args.lr,
        betas=(args.b1, args.b2),
        eps=args.eps,
    )

    return {
        "G_forward_da":    G_forward_da,
        "G_forward_pred":  G_forward_pred,
        "D_forward":       D_forward,
        "feature_extractor": feature_extractor,
        "criterion_GAN":   criterion_GAN,
        "criterion_content": criterion_content,
        "TV_Loss":         TV_Loss_fn,
        "G_optim":         G_optim,
        "D_optim":         D_optim,
        "cuda":            cuda,
        "device":          device,
    }
