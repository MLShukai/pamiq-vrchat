from typing import override

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Distribution

from .components.fc_scalar_head import FCScalarHead
from .components.multi_discretes import FCMultiCategoricalHead
from .components.qlstm import QLSTM


class PolicyValueCommon(nn.Module):
    """Module with shared models for policy and value functions."""

    def __init__(
        self,
        obs_dim: int,
        action_choices: list[int],
        dim: int,
        depth: int,
        dim_ff_hidden: int,
        dropout: float = 0.1,
    ) -> None:
        """Constructs the PolicyValueCommon network.

        Args:
            obs_dim: Dimension of the observation input.
            depth: Number of recurrent layers in the core QLSTM model.
            dim: Hidden dimension of the model.
            dim_ff_hidden: Hidden dimension of the feed-forward networks in QLSTM.
            dropout: Dropout rate for regularization.
            action_choices: List of integers specifying the number of choices for
        """
        super().__init__()
        self.obs_proj = nn.Linear(obs_dim, dim) if obs_dim != dim else nn.Identity()
        self.core_model = QLSTM(depth, dim, dim_ff_hidden, dropout)
        self.policy_head = FCMultiCategoricalHead(dim, action_choices)
        self.value_head = FCScalarHead(dim, squeeze_scalar_dim=True)

    @override
    def forward(
        self, observation: Tensor, hidden: Tensor
    ) -> tuple[Distribution, Tensor, Tensor]:
        """Process observation and compute policy and value outputs.

        Args:
            observation: Input observation tensor
            hidden: Hidden state tensor from previous timestep

        Returns:
            A tuple containing:
                - Distribution representing the policy (action probabilities)
                - Tensor containing the estimated state value
                - Updated hidden state tensor for use in next forward pass
        """
        obs_embed = self.obs_proj(observation)
        x, hidden = self.core_model(obs_embed, hidden)
        return self.policy_head(x), self.value_head(x), hidden

    @override
    def __call__(
        self, observation: Tensor, hidden: Tensor
    ) -> tuple[Distribution, Tensor, Tensor]:
        """Override __call__ with proper type annotations."""
        return super().__call__(observation, hidden)
