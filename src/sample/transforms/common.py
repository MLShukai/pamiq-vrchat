from typing import override

import torch
import torch.nn as nn


class Standardize(nn.Module):
    """Standardize input by removing mean and dividing by standard deviation.

    This module performs standardization (zero mean, unit variance) on
    the input tensor.
    """

    def __init__(self, eps: float = 1e-8) -> None:
        """Initialize the Standardize transform.

        Args:
            eps: Small value to add to standard deviation to avoid division by zero.
        """
        super().__init__()
        self.eps = eps

    @override
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Apply standardization to the input tensor.

        Args:
            input: Input tensor to standardize.

        Returns:
            Standardized tensor with zero mean and unit variance.
        """
        if input.numel() <= 1:  # Can not compute std
            return torch.zeros_like(input)

        return (input - input.mean()) / (input.std() + self.eps)
