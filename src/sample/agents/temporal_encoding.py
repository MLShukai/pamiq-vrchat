from pathlib import Path
from typing import override

import torch
from pamiq_core import Agent
from tensordict import TensorDict
from torch import Tensor

from sample.data import BufferName, DataKey
from sample.models import ModelName


class TemporalEncodingAgent(Agent[dict[str, Tensor], Tensor]):
    """Agent that encodes multimodal temporal observations using a recurrent
    model.

    This agent processes sequences of multimodal observations (e.g.,
    image and audio) through a temporal encoder with hidden state,
    maintaining context across timesteps. It collects both observations
    and hidden states for potential training.
    """

    @override
    def __init__(self, initial_hidden: Tensor):
        """Initialize the TemporalEncodingAgent.

        Args:
            initial_hidden: Initial hidden state tensor for the temporal encoder
        """
        super().__init__()
        self.encoder_hidden_state = initial_hidden

    @override
    def on_inference_models_attached(self) -> None:
        """Set up the temporal encoder model when inference models are
        attached."""
        super().on_inference_models_attached()
        self.encoder = self.get_inference_model(ModelName.TEMPORAL_ENCODER)

    @override
    def on_data_collectors_attached(self) -> None:
        """Set up the data collector when data collectors are attached."""
        super().on_data_collectors_attached()
        self.collector = self.get_data_collector(BufferName.TEMPORAL)

    @override
    def step(self, observation: dict[str, Tensor] | TensorDict) -> Tensor:
        """Process multimodal observations and return temporal encoding.

        Updates the internal hidden state and collects observation/hidden data.

        Args:
            observation: Dictionary mapping modality names to observation tensors

        Returns:
            Temporal encoding of the observation
        """
        obs = TensorDict(observation)
        self.collector.collect(
            {
                DataKey.OBSERVATION: obs.cpu(),
                DataKey.HIDDEN: self.encoder_hidden_state.cpu(),
            }
        )
        out, self.encoder_hidden_state = self.encoder(obs, self.encoder_hidden_state)
        return out

    @override
    def save_state(self, path: Path) -> None:
        """Save agent state including the hidden state tensor.

        Args:
            path: Directory path where to save the state
        """
        super().save_state(path)
        path.mkdir(exist_ok=True)
        torch.save(self.encoder_hidden_state, path / "encoder_hidden_state.pt")

    @override
    def load_state(self, path: Path) -> None:
        """Load agent state including the hidden state tensor.

        Args:
            path: Directory path from where to load the state
        """
        super().load_state(path)
        self.encoder_hidden_state = torch.load(
            path / "encoder_hidden_state.pt",
            map_location=self.encoder_hidden_state.device,
        )
