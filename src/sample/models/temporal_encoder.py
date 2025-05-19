from collections.abc import Mapping
from typing import NamedTuple, override

import torch
import torch.nn as nn
import torch.nn.functional as F
from pamiq_core.torch import TorchTrainingModel, get_device
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


class ObsInfo(NamedTuple):
    """Configuration for observation processing in temporal encoder.

    This named tuple defines the dimensions and feature stack configuration for
    each modality processed by the temporal encoder.

    Attributes:
        dim_in: Input dimension of the observation.
        dim_out: Output dimension after feature transformation.
        num_tokens: Number of tokens of observation.
    """

    dim_in: int
    dim_out: int
    num_tokens: int


def instantiate(
    obs_infos: Mapping[str, ObsInfo | tuple[int, int, int]],
    depth: int,
    dim: int,
    dim_ff_hidden: int,
    dropout: float,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> TorchTrainingModel[TemporalEncoder]:
    """Create a temporal encoder model with the specified configuration.

    This factory function instantiates a TemporalEncoder model with stacked feature
    processing for each modality (expected JEPA output.), and wraps it in a TorchTrainingModel for use with
    the PAMIQ-Core framework.

    Args:
        obs_infos: Dictionary mapping modality names to their observation configuration.
        depth: Number of recurrent layers in the core QLSTM model.
        dim: Hidden dimension of the encoder.
        dim_ff_hidden: Hidden dimension of the feed-forward networks in QLSTM.
        dropout: Dropout rate for regularization.
        device: PyTorch device to place the model on. Defaults to None (uses current device).
        dtype: PyTorch data type for the model parameters. Defaults to None (uses default dtype).

    Returns:
        TorchTrainingModel containing the configured temporal encoder.
    """
    from .components.deterministic_normal import FCDeterministicNormalHead
    from .components.qlstm import QLSTM
    from .components.stacked_features import LerpStackedFeatures, ToStackedFeatures

    obs_flattens = {}
    obs_hat_heads = {}
    flattened_size = 0
    for name, info in obs_infos.items():
        info = ObsInfo(*info)
        obs_flattens[name] = LerpStackedFeatures(
            info.dim_in, info.dim_out, info.num_tokens
        )

        obs_hat_heads[name] = nn.Sequential(
            ToStackedFeatures(dim, info.dim_in, info.num_tokens),
            FCDeterministicNormalHead(info.dim_in, info.dim_in),
        )
        flattened_size += info.dim_out

    encoder = TemporalEncoder(
        obs_flattens,
        nn.Linear(flattened_size, dim),
        QLSTM(depth, dim, dim_ff_hidden, dropout),
        obs_hat_heads,
    )
    return TorchTrainingModel(
        encoder,
        has_inference_model=True,
        inference_procedure=TemporalEncoder.infer,
        device=device,
        dtype=dtype,
    )
