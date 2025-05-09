from typing import override

import torch.nn as nn
from torch import Tensor
from torch.distributions import Distribution


class PolicyValueCommon(nn.Module):
    """Module with shared models for policy and value functions."""

    def __init__(
        self,
        observation_flatten: nn.Module,
        core_model: nn.Module,
        policy_head: nn.Module,
        value_head: nn.Module,
    ) -> None:
        """Constructs the model with components.

        Args:
            observation_flatten: Layer that processes observation.
            core_model: Layer that processes observation with hidden state
            policy_head: Layer that predicts actions.
            value_head: Layer that predicts state values.
        """
        super().__init__()
        self.observation_flatten = observation_flatten
        self.core_model = core_model
        self.policy_head = policy_head
        self.value_head = value_head

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
        obs_embed = self.observation_flatten(observation)
        x, hidden = self.core_model(obs_embed, hidden)
        return self.policy_head(x), self.value_head(x), hidden

    @override
    def __call__(
        self, observation: Tensor, hidden: Tensor
    ) -> tuple[Distribution, Tensor, Tensor]:
        """Override __call__ with proper type annotations."""
        return super().__call__(observation, hidden)
