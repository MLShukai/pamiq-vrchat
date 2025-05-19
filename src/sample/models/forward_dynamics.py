from typing import override

import torch
import torch.nn as nn
from pamiq_core.torch import TorchTrainingModel
from torch import Tensor
from torch.distributions import Distribution

from .components.stacked_hidden_state import StackedHiddenState


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
        observation_flatten: nn.Module,
        action_flatten: nn.Module,
        obs_action_projection: nn.Module,
        core_model: StackedHiddenState,
        obs_hat_dist_head: nn.Module,
    ) -> None:
        """Initialize the forward dynamics model.

        Args:
            observation_flatten: Module to flatten observation tensors.
            action_flatten: Module to flatten action tensors.
            obs_action_projection: Module to project concatenated observation and action.
            core_model: Recurrent core model maintaining hidden state.
            obs_hat_dist_head: Module converting features to observation distribution.
        """
        super().__init__()
        self.observation_flatten = observation_flatten
        self.action_flatten = action_flatten
        self.obs_action_projection = obs_action_projection
        self.core_model = core_model
        self.obs_hat_dist_head = obs_hat_dist_head

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
        obs_flat = self.observation_flatten(obs)
        action_flat = self.action_flatten(action)
        x = self.obs_action_projection(torch.cat((obs_flat, action_flat), dim=-1))
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


def instantiate(
    obs_dim: int,
    action_choices: list[int],
    action_dim: int,
    depth: int,
    dim: int,
    dim_ff_hidden: int,
    dropout: float,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> TorchTrainingModel[ForwardDynamics]:
    """Create a forward dynamics model with the specified configuration.

    This factory function instantiates a ForwardDynamics model that predicts the next
    observation given the current observation and action. It uses a QLSTM core for
    temporal encoding and wraps the model in a TorchTrainingModel for use with the
    PAMIQ-Core framework.

    Args:
        obs_dim: Dimension of the observation input.
        action_choices: List specifying the number of choices for each discrete action dimension.
        action_dim: Embedding dimension for each action component.
        depth: Number of recurrent layers in the core QLSTM model.
        dim: Hidden dimension of the model.
        dim_ff_hidden: Hidden dimension of the feed-forward networks in QLSTM.
        dropout: Dropout rate for regularization.
        device: PyTorch device to place the model on. Defaults to None (uses current device).
        dtype: PyTorch data type for the model parameters. Defaults to None (uses default dtype).

    Returns:
        TorchTrainingModel containing the configured forward dynamics model.
    """

    from .components.deterministic_normal import FCDeterministicNormalHead
    from .components.multi_discretes import MultiEmbeddings
    from .components.qlstm import QLSTM

    model = ForwardDynamics(
        observation_flatten=nn.Identity(),
        action_flatten=MultiEmbeddings(action_choices, action_dim, do_flatten=True),
        obs_action_projection=nn.Linear(
            obs_dim + action_dim * len(action_choices), dim
        ),
        core_model=QLSTM(depth, dim, dim_ff_hidden, dropout),
        obs_hat_dist_head=FCDeterministicNormalHead(
            dim, obs_dim, squeeze_feature_dim=False
        ),
    )
    return TorchTrainingModel(
        model, has_inference_model=True, device=device, dtype=dtype
    )
