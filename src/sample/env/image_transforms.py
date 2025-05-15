import math
from collections.abc import Callable
from typing import override

import torch
import torch.nn as nn
import torchvision.transforms.v2.functional as F
from torch import Tensor
from torchvision.transforms.v2 import (
    Compose,
    ConvertImageDtype,
    ToImage,
    ToPureTensor,
)

from pamiq_vrchat.sensors.image import ImageFrame


class ResizeAndCenterCrop(nn.Module):
    """Resize and center crop transform for images.

    This module resizes the input image to fit within the target size
    while maintaining aspect ratio, then performs a center crop to the
    exact target size.
    """

    def __init__(self, size: tuple[int, int]) -> None:
        """Initialize the ResizeAndCenterCrop transform.

        Args:
            size: Target size as (height, width) tuple.
        """
        super().__init__()
        self.size = list(size)

    @override
    def forward(self, input: Tensor) -> Tensor:
        """Apply resize and center crop to the input tensor.

        Args:
            input: Input tensor with shape (..., H, W) where H, W are height and width.

        Returns:
            Transformed tensor with shape (..., size[0], size[1]).

        Raises:
            ValueError: If input has less than 3 dimensions.
            ValueError: If input height or width is 0.
        """
        if input.ndim < 3:
            raise ValueError(
                f"Input tensor must have at least 3 dimensions, got {input.ndim}"
            )

        inpt_img_size = input.shape[-2:]
        if min(inpt_img_size) == 0:
            raise ValueError(
                f"Input image dimensions must be non-zero, got {tuple(inpt_img_size)}"
            )

        max_size = math.ceil(max(inpt_img_size) / min(inpt_img_size) * max(self.size))
        input = F.resize(input, None, max_size=max_size)
        input = F.center_crop(input, self.size)
        return input


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
    def forward(self, input: Tensor) -> Tensor:
        """Apply standardization to the input tensor.

        Args:
            input: Input tensor to standardize.

        Returns:
            Standardized tensor with zero mean and unit variance.
        """
        if input.numel() <= 1:  # Can not compute std
            return torch.zeros_like(input)

        return (input - input.mean()) / (input.std() + self.eps)


def create_vrchat_transform(
    size: tuple[int, int], dtype: torch.dtype = torch.float
) -> Callable[[ImageFrame], Tensor]:
    """Create a composed transform for VRChat image processing.

    Creates a transform pipeline that:
    1. Converts image to tensor
    2. Converts dtype as specified
    3. Resizes and center crops to target size
    4. Standardizes values
    5. Removes any image metadata

    Args:
        size: Target size as (height, width) tuple.
        dtype: Target dtype for the tensor. Defaults to torch.float.

    Returns:
        A callable that transforms ImageFrame to standardized Tensor.
    """
    transform = Compose(
        [
            ToImage(),
            ResizeAndCenterCrop(size),
            ConvertImageDtype(dtype),
            Standardize(),
            ToPureTensor(),
        ]
    )
    return transform
