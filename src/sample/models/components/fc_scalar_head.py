from typing import override

import torch.nn as nn
from torch import Tensor


class FCScalarHead(nn.Module):
    """Fully connected layer that outputs a scalar value.

    This module applies a linear transformation to the input features
    and outputs a single scalar value per input feature vector. The
    output can optionally have its last dimension squeezed.
    """

    def __init__(self, dim_in: int, squeeze_scalar_dim: bool = False) -> None:
        """Initialize the fully connected scalar head.

        Args:
            dim_in: Number of input features.
            squeeze_scalar_dim: If True, squeeze the last dimension of the output.
        """
        super().__init__()
        self.fc = nn.Linear(dim_in, 1)
        self.squeeze_scalar_dim = squeeze_scalar_dim

    @override
    def forward(self, x: Tensor) -> Tensor:
        """Calculate scalar output from input features.

        Args:
            x: Input tensor of shape (..., dim_in).

        Returns:
            Output tensor of shape (..., 1) if squeeze_scalar_dim is False,
            or shape (...) if squeeze_scalar_dim is True.
        """
        out: Tensor = self.fc(x)
        if self.squeeze_scalar_dim:
            out = out.squeeze(-1)
        return out
