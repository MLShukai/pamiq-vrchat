import math
from collections.abc import Callable
from typing import override

import torch
import torch.nn as nn
import torchvision.transforms.v2.functional as F
from torch import Tensor
from torchvision.transforms.v2 import (
    Compose,
    ToDtype,
    ToImage,
    ToPureTensor,
)

from pamiq_vrchat.sensors.image import ImageFrame

from .common import Standardize


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

        input_img_size = input.shape[-2:]
        if min(input_img_size) == 0:
            raise ValueError(
                f"Input image dimensions must be non-zero, got {tuple(input_img_size)}"
            )

        ar_input = input_img_size[1] / input_img_size[0]
        ar_size = self.size[1] / self.size[0]

        if ar_input < ar_size:
            scale_size = (math.ceil(self.size[1] / ar_input), self.size[1])
        else:
            scale_size = (self.size[0], math.ceil(self.size[0] * ar_input))
        input = F.resize(input, list(scale_size))
        input = F.center_crop(input, self.size)
        return input


def create_transform(
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
            ToDtype(dtype, scale=True),
            ResizeAndCenterCrop(size),
            Standardize(),
            ToPureTensor(),
        ]
    )
    return transform
