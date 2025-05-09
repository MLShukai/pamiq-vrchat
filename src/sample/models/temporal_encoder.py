from collections.abc import Mapping
from typing import override

import torch
import torch.nn as nn
import torch.nn.functional as F
from pamiq_core.torch import get_device
from torch import Tensor
from torch.distributions import Distribution

from .components.stacked_hidden_state import StackedHiddenState


class TemporalEncoder(nn.Module):
    """Multimodal temporal encoder framework module."""

    def __init__(
        self,
        observation_flattens: Mapping[str, nn.Module],
        flattened_obses_projection: nn.Module,
        core_model: StackedHiddenState,
        obs_hat_dist_heads: Mapping[str, nn.Module],
    ) -> None:
        """Initializes TemporalEncoder.

        Args:
            observation_flattens: The dict contains the flatten module for each modality
            flattened_obses_projection: Projects the concatenated flattened observations to fixed size vector.
            core_model: Core model for temporal encoding.
            obs_hat_dist_heads: Observation prediction heads for each modality

        Raises:
            KeyError: If keys between observation_flattens and obs_hat_dist_heads are not matched.
        """
        super().__init__()
        if obs_hat_dist_heads.keys() != observation_flattens.keys():
            raise KeyError(
                "The keys between observation_flattens and obs_hat_dist_heads are not matched! "
                f"observation_flattens keys: {observation_flattens.keys()}, "
                f"obs_hat_dist_heads keys: {obs_hat_dist_heads.keys()}"
            )

        self.observation_flattens = nn.ModuleDict(observation_flattens)

        self.flattened_obses_projection = flattened_obses_projection
        self.core_model = core_model
        self.obs_hat_dist_heads = nn.ModuleDict(obs_hat_dist_heads)

    @override
    def forward(
        self, observations: Mapping[str, Tensor], hidden: Tensor
    ) -> tuple[Tensor, Tensor, Mapping[str, Distribution]]:
        """Forward path of multimodal temporal encoder.

        Args:
            observations: Dictionary mapping modality names to observation tensors
            hidden: Hidden state tensor for temporal module

        Returns:
            A tuple containing:
                - Embedded observation representation
                - Next hidden state
                - Dictionary of predicted observation distributions for each modality

        Raises:
            KeyError: If keys between observation_flattens and observations are not matched.
        """
        if observations.keys() != self.observation_flattens.keys():
            raise KeyError("Observations keys are not matched!")

        obs_flats = [
            layer(observations[k]) for k, layer in self.observation_flattens.items()
        ]

        x = self.flattened_obses_projection(torch.cat(obs_flats, dim=-1))
        x, next_hidden = self.core_model(x, hidden)
        obs_hat_dists = {k: layer(x) for k, layer in self.obs_hat_dist_heads.items()}
        return x, next_hidden, obs_hat_dists

    @override
    def __call__(
        self, observations: Mapping[str, Tensor], hidden: Tensor
    ) -> tuple[Tensor, Tensor, Mapping[str, Distribution]]:
        """Call forward method with type checking.

        This method is an override of nn.Module.__call__ to provide
        proper type hints. It delegates to the forward method.
        """
        return super().__call__(observations, hidden)

    def infer(
        self, observations: Mapping[str, Tensor], hidden: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Perform inference without generating observation predictions.

        Args:
            observations: Dictionary mapping modality names to observation tensors
            hidden: Hidden state tensor for temporal module

        Returns:
            A tuple containing:
                - Layer-normalized embedded observation representation
                - Next hidden state
        """
        device = get_device(self)
        observations = {k: v.to(device) for k, v in observations.items()}
        x, next_hidden, _ = self(observations, hidden.to(device))
        x = F.layer_norm(x, x.shape[-1:])
        return x, next_hidden
