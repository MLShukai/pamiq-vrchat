from collections.abc import Mapping
from dataclasses import dataclass
from typing import override

import torch
import torch.nn as nn
import torch.nn.functional as F
from pamiq_core.torch import get_device
from torch import Tensor
from torch.distributions import Distribution

from .components.deterministic_normal import FCDeterministicNormalHead
from .components.qlstm import QLSTM
from .components.stacked_features import LerpStackedFeatures, ToStackedFeatures


@dataclass
class ObsInfo:
    """Configuration for observation processing in temporal encoder.

    This dataclass defines the dimensions and feature stack configuration for
    each modality processed by the temporal encoder.

    Attributes:
        dim: Input dimension of the observation.
        dim_hidden: Hidden dimension after feature transformation.
        num_tokens: Number of tokens of observation.
    """

    dim: int
    dim_hidden: int
    num_tokens: int


class TemporalEncoder(nn.Module):
    """Multimodal temporal encoder framework module."""

    def __init__(
        self,
        obs_infos: Mapping[str, ObsInfo | tuple[int, int, int]],
        dim: int,
        depth: int,
        dim_ff_hidden: int,
        dropout: float,
    ) -> None:
        """Initializes TemporalEncoder.

        Args:
            obs_infos: Dictionary mapping modality names to their observation configuration.
            depth: Number of recurrent layers in the core QLSTM model.
            dim: Hidden dimension of the encoder.
            dim_ff_hidden: Hidden dimension of the feed-forward networks in QLSTM.
            dropout: Dropout rate for regularization.
        """
        super().__init__()
        obs_flattens = {}
        obs_hat_heads = {}
        flattened_size = 0
        for name, info in obs_infos.items():
            if isinstance(info, tuple):
                info = ObsInfo(*info)
            obs_flattens[name] = LerpStackedFeatures(
                info.dim, info.dim_hidden, info.num_tokens
            )

            obs_hat_heads[name] = nn.Sequential(
                ToStackedFeatures(dim, info.dim, info.num_tokens),
                FCDeterministicNormalHead(info.dim, info.dim),
            )
            flattened_size += info.dim_hidden

        self.observation_flattens = nn.ModuleDict(obs_flattens)

        self.flattened_obses_projection = nn.Linear(flattened_size, dim)
        self.core_model = QLSTM(depth, dim, dim_ff_hidden, dropout)
        self.obs_hat_dist_heads = nn.ModuleDict(obs_hat_heads)

    def _common_flow(
        self, observations: Mapping[str, Tensor], hidden: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Common data flow of Temporal Encoder."""
        if observations.keys() != self.observation_flattens.keys():
            raise KeyError("Observations keys are not matched!")

        obs_flats = [
            layer(observations[k]) for k, layer in self.observation_flattens.items()
        ]

        x = self.flattened_obses_projection(torch.cat(obs_flats, dim=-1))
        x, next_hidden = self.core_model(x, hidden)
        return x, next_hidden

    @override
    def forward(
        self, observations: Mapping[str, Tensor], hidden: Tensor
    ) -> tuple[Mapping[str, Distribution], Tensor]:
        """Forward path of multimodal temporal encoder.

        Args:
            observations: Dictionary mapping modality names to observation tensors
            hidden: Hidden state tensor for temporal module

        Returns:
            A tuple containing:
                - Dictionary of predicted observation distributions for each modality
                - Next hidden state

        Raises:
            KeyError: If keys between observation_flattens and observations are not matched.
        """
        x, next_hidden = self._common_flow(observations, hidden)
        obs_hat_dists = {k: layer(x) for k, layer in self.obs_hat_dist_heads.items()}
        return obs_hat_dists, next_hidden

    @override
    def __call__(
        self, observations: Mapping[str, Tensor], hidden: Tensor
    ) -> tuple[Mapping[str, Distribution], Tensor]:
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
        x, next_hidden = self._common_flow(observations, hidden.to(device))
        x = F.layer_norm(x, x.shape[-1:])
        return x, next_hidden
