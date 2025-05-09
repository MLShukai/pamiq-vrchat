import math
from collections.abc import Iterable
from typing import override

import torch
import torch.nn as nn
from torch import Size, Tensor
from torch.distributions import Normal

SHIFT_ZERO = 1.0 / math.sqrt(2.0 * math.pi)
SCALE_ONE = 1.0 / math.sqrt(2.0)


class DeterministicNormal(Normal):
    """Always samples only mean.

    A specialized Normal distribution that always returns the mean value
    when sampling, making it deterministic. This is useful for
    deterministic forward dynamics prediction.
    """

    @override
    def sample(self, sample_shape: Iterable[int] = Size()) -> Tensor:
        return self.rsample(sample_shape)

    @override
    def rsample(self, sample_shape: Iterable[int] = Size()) -> Tensor:
        shape = self._extended_shape(Size(sample_shape))
        return self.mean.expand(shape)


class FCDeterministicNormalHead(nn.Module):
    """The layer which returns the normal distribution with fixed standard
    deviation.

    This module applies a linear transformation to the input features and returns a
    DeterministicNormal distribution with the transformed values as the mean and a
    fixed standard deviation.

    See: https://github.com/MLShukai/ami/issues/117
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        std: float = SCALE_ONE,
        squeeze_feature_dim: bool = False,
    ) -> None:
        """Initialize the FCDeterministicNormalHead layer.

        Args:
            dim_in: Number of input features.
            dim_out: Number of output features representing the dimensionality of the distribution mean.
            std: Fixed standard deviation value for all dimensions of the distribution.
                Defaults to 1/sqrt(2), which matches MSE gradient.
            squeeze_feature_dim: If True, removes the last dimension of the output.
                Only valid when dim_out=1. Useful when the output should not have
                a trailing singleton dimension.

        Raises:
            ValueError: If squeeze_feature_dim is True but dim_out is not 1.
        """
        super().__init__()

        if squeeze_feature_dim and dim_out != 1:
            raise ValueError("Can not squeeze feature dimension!")

        self.fc = nn.Linear(dim_in, dim_out)
        self.std = std
        self.squeeze_feature_dim = squeeze_feature_dim

    @override
    def forward(self, x: Tensor) -> DeterministicNormal:
        """Compute the normal distribution with fixed standard deviation from
        the input.

        Args:
            x: Input tensor of shape [..., dim_in].

        Returns:
            A DeterministicNormal distribution with mean from the linear transformation
            and fixed standard deviation.
        """
        mean: Tensor = self.fc(x)
        if self.squeeze_feature_dim:
            mean = mean.squeeze(-1)
        std = torch.full_like(mean, self.std)
        return DeterministicNormal(mean, std)
