from typing import override

import torch
import torch.nn as nn
from torch import Tensor


class LerpStackedFeatures(nn.Module):
    """Linear interpolation along stack of features.

    This module performs linear interpolation between stacked features
    using learned coefficients. The interpolation is performed across
    the stack dimension using a softmax-weighted combination.
    """

    def __init__(self, dim_in: int, dim_out: int, num_stack: int) -> None:
        """Initialize the linear interpolation module.

        Args:
            dim_in: Input feature dimension for each stack element.
            dim_out: Output feature dimension after interpolation.
            num_stack: Number of features in the stack to interpolate between.
        """
        super().__init__()

        self.dim_in = dim_in
        self.dim_out = dim_out
        self.num_stack = num_stack

        self.feature_linear_weight = nn.Parameter(
            torch.randn(num_stack, dim_in, dim_out) * (dim_out**-0.5)
        )
        self.feature_linear_bias = nn.Parameter(
            torch.randn(num_stack, dim_out) * (dim_out**-0.5)
        )
        self.logit_coef_proj = nn.Linear(num_stack * dim_in, num_stack)

    @override
    def forward(self, stacked_features: Tensor) -> Tensor:
        """Perform linear interpolation across stacked features.

        Args:
            stacked_features: Input tensor of shape (*, num_stack, dim_in) where
                * can be any number of batch dimensions.

        Returns:
            Interpolated features of shape (*, dim_out).
        """
        no_batch = len(stacked_features.shape) == 2
        if no_batch:
            stacked_features = stacked_features.unsqueeze(0)

        batch_shape = stacked_features.shape[:-2]
        n_stack, dim = stacked_features.shape[-2:]
        stacked_features = stacked_features.reshape(-1, n_stack, dim)
        batch = stacked_features.size(0)

        logit_coef = self.logit_coef_proj(
            stacked_features.reshape(batch, n_stack * dim)
        )

        feature_linear = torch.einsum(
            "sio,bsi->bso", self.feature_linear_weight, stacked_features
        ) + self.feature_linear_bias.unsqueeze(0)

        out = torch.einsum(
            "bs,bsi->bi", torch.softmax(logit_coef, dim=-1), feature_linear
        )

        out = out.reshape(*batch_shape, -1)

        if no_batch:
            out = out.squeeze(0)
        return out


class ToStackedFeatures(nn.Module):
    """Convert input features to stacked features.

    This module transforms input features into a stack of feature
    representations through learned linear transformations.
    """

    def __init__(self, dim_in: int, dim_out: int, num_stack: int) -> None:
        """Initialize the feature stacking module.

        Args:
            dim_in: Input feature dimension.
            dim_out: Output feature dimension for each stack element.
            num_stack: Number of features to produce in the stack.
        """
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(dim_in, num_stack, dim_out) * (dim_out**-0.5)
        )
        self.bias = nn.Parameter(torch.randn(num_stack, dim_out) * (dim_out**-0.5))

        self.num_stack = num_stack

    @override
    def forward(self, feature: Tensor) -> Tensor:
        """Convert input features to stacked representation.

        Args:
            feature: Input tensor of shape (*, dim_in) where * can be
                any number of batch dimensions.

        Returns:
            Stacked features of shape (*, num_stack, dim_out).
        """
        no_batch = feature.ndim == 1
        if no_batch:
            feature = feature.unsqueeze(0)
        batch_shape = feature.shape[:-1]
        feature = feature.reshape(-1, feature.size(-1))

        out = torch.einsum("bi,isj->bsj", feature, self.weight) + self.bias
        out = out.reshape(*batch_shape, *out.shape[-2:])

        if no_batch:
            out = out.squeeze(0)
        return out
