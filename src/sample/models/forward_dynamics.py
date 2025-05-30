from typing import override

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Distribution

from .components.deterministic_normal import FCDeterministicNormalHead
from .components.multi_discretes import MultiEmbeddings
from .components.qlstm import QLSTM


class ForwardDynamics(nn.Module):
    """Forward dynamics model predicting next observation distribution given
    current observation and action.

    This model combines observation and action data through a series of
    transformations to predict the distribution of the next observation.
    It uses a core recurrent model to maintain hidden state across
    sequential predictions.
    """

    @override
    def __init__(
        self,
        obs_dim: int,
        action_choices: list[int],
        action_dim: int,
        dim: int,
        depth: int,
        dim_ff_hidden: int,
        dropout: float = 0.1,
    ) -> None:
        """Initialize the forward dynamics model.

        Args:
            obs_dim: Dimension of the observation input.
            action_choices: List specifying the number of choices for each discrete action dimension.
            action_dim: Embedding dimension for each action component.
            dim: Hidden dimension of the model.
            depth: Number of recurrent layers in the core QLSTM model.
            dim_ff_hidden: Hidden dimension of the feed-forward networks in QLSTM.
            dropout: Dropout rate for regularization.
        """
        super().__init__()
        self.action_flatten = MultiEmbeddings(
            action_choices, action_dim, do_flatten=True
        )
        self.obs_action_projection = nn.Linear(
            obs_dim + action_dim * len(action_choices), dim
        )
        self.core_model = QLSTM(depth, dim, dim_ff_hidden, dropout)
        self.obs_hat_dist_head = FCDeterministicNormalHead(dim, obs_dim)

    @override
    def forward(
        self, obs: Tensor, action: Tensor, hidden: Tensor
    ) -> tuple[Distribution, Tensor]:
        """Forward pass to predict next observation distribution.

        Args:
            obs: Current observation tensor.
            action: Action tensor.
            hidden: Hidden state from previous timestep.

        Returns:
            A tuple containing:
                - Distribution representing predicted next observation.
                - Updated hidden state tensor for use in next prediction.
        """
        action_flat = self.action_flatten(action)
        x = self.obs_action_projection(torch.cat((obs, action_flat), dim=-1))
        x, next_hidden = self.core_model(x, hidden)
        obs_hat_dist = self.obs_hat_dist_head(x)
        return obs_hat_dist, next_hidden

    @override
    def __call__(
        self, obs: Tensor, action: Tensor, hidden: Tensor
    ) -> tuple[Distribution, Tensor]:
        """Override __call__ with proper type annotations.

        See forward() method for full documentation.
        """
        return super().__call__(obs, action, hidden)
