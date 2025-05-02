"""Utility tools for model definitions."""

import torch.nn as nn


def init_weights(m: nn.Module, init_std: float) -> None:
    """Initialize the weights with truncated normal distribution and zeros for
    biases.

    Args:
        m: Module to initialize.
        init_std: Standard deviation for the truncated normal initialization.
    """
    match m:
        case nn.Linear() | nn.Conv2d() | nn.ConvTranspose2d():
            nn.init.trunc_normal_(m.weight, std=init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        case nn.LayerNorm():
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        case _:
            pass
