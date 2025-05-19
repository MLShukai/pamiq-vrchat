from typing import override

import torch
import torch.nn as nn
from pamiq_core.torch import TorchTrainingModel
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


def instantiate(
    obs_dim: int,
    depth: int,
    dim: int,
    dim_ff_hidden: int,
    dropout: float,
    action_choices: list[int],
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> TorchTrainingModel[PolicyValueCommon]:
    """Create a policy-value model with the specified configuration.

    This factory function instantiates a PolicyValueCommon model with QLSTM core,
    multi-categorical policy head, and scalar value head, then wraps it in a
    TorchTrainingModel for use with the PAMIQ-Core framework.

    Args:
        obs_dim: Dimension of the observation input.
        depth: Number of recurrent layers in the core QLSTM model.
        dim: Hidden dimension of the model.
        dim_ff_hidden: Hidden dimension of the feed-forward networks in QLSTM.
        dropout: Dropout rate for regularization.
        action_choices: List of integers specifying the number of choices for each
            discrete action dimension.
        device: PyTorch device to place the model on. Defaults to None (uses current device).
        dtype: PyTorch data type for the model parameters. Defaults to None (uses default dtype).

    Returns:
        TorchTrainingModel containing the configured policy-value model.
    """
    from .components.fc_scalar_head import FCScalarHead
    from .components.multi_discretes import FCMultiCategoricalHead
    from .components.qlstm import QLSTM

    model = PolicyValueCommon(
        observation_flatten=nn.Linear(obs_dim, dim)
        if obs_dim != dim
        else nn.Identity(),
        core_model=QLSTM(depth, dim, dim_ff_hidden, dropout),
        policy_head=FCMultiCategoricalHead(dim, action_choices),
        value_head=FCScalarHead(dim, True),
    )

    return TorchTrainingModel(
        model,
        has_inference_model=True,
        device=device,
        dtype=dtype,
    )
