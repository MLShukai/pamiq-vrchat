from typing import override

import torch
import torch.nn as nn
from torch import Tensor


class LerpStackedFeatures(nn.Module):
    """Linear interpolation along stack of features.

    This module performs linear interpolation between stacked features
    using learned coefficients. The interpolation is performed across
    the stack dimension using a softmax-weighted combination.

    The weights are initialized using Xavier initialization to maintain
    gradient flow, while the interpolation coefficients start with small
    random values to break symmetry.
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
            torch.empty(num_stack, dim_in, dim_out)
        )
        self.feature_linear_bias = nn.Parameter(torch.empty(num_stack, dim_out))
        self.logit_coef_proj = nn.Linear(num_stack * dim_in, num_stack)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize module weights using appropriate strategies.

        Uses Xavier uniform initialization for feature transformations
        to maintain gradient magnitudes across layers, zero
        initialization for biases, and small random values for
        interpolation coefficients to break initial symmetry.
        """
        # Xavier uniform initialization for feature transformation weights
        nn.init.xavier_uniform_(self.feature_linear_weight)

        # Zero initialization for feature transformation biases
        nn.init.zeros_(self.feature_linear_bias)

        # Small random initialization for interpolation coefficients
        # This ensures different stacks contribute differently from the start
        nn.init.normal_(self.logit_coef_proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.logit_coef_proj.bias)

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
    representations through learned linear transformations. Each stack
    element learns a different representation of the input, allowing for
    diverse feature extraction.

    Weights are initialized using Xavier initialization to ensure stable
    gradient flow across different stack elements.
    """

    def __init__(self, dim_in: int, dim_out: int, num_stack: int) -> None:
        """Initialize the feature stacking module.

        Args:
            dim_in: Input feature dimension.
            dim_out: Output feature dimension for each stack element.
            num_stack: Number of features to produce in the stack.
        """
        super().__init__()

        self.weight = nn.Parameter(torch.empty(dim_in, num_stack, dim_out))
        self.bias = nn.Parameter(torch.empty(num_stack, dim_out))

        self.num_stack = num_stack

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize module weights using Xavier initialization.

        Xavier initialization helps maintain consistent gradient
        magnitudes throughout the network by considering both input and
        output dimensions. Biases are initialized to zero following
        standard practice.
        """
        # Xavier uniform initialization for weights
        # This considers fan_in=dim_in and fan_out=dim_out for each stack
        for i in range(self.num_stack):
            nn.init.xavier_uniform_(self.weight[:, i, :])

        # Zero initialization for biases
        nn.init.zeros_(self.bias)

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
