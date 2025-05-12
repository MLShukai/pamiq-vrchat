from typing import override

from pamiq_core import Agent
from torch import Tensor

from sample.data import DataKey


class UnimodalEncodingAgent(Agent[Tensor, Tensor]):
    """Agent that encodes observations into embeddings using a pre-trained
    encoder model.

    This agent processes observations (such as images or audio) through
    an encoder model and returns the encoded representation. It also
    stores the original observations in a data buffer for potential
    training purposes.
    """

    @override
    def __init__(self, model_name: str, data_collector_name: str) -> None:
        """Initialize the UnimodalEncodingAgent.

        Args:
            model_name: Name of the encoder model to use
            data_collector_name: Name of the data collector to store observations
        """
        super().__init__()
        self.model_name = model_name
        self.data_collector_name = data_collector_name

    @override
    def on_inference_models_attached(self) -> None:
        """Set up the encoder model when inference models are attached.

        This method is called automatically by the PAMIQ framework when
        inference models are attached to the agent.
        """
        super().on_inference_models_attached()
        self.encoder = self.get_inference_model(self.model_name)

    @override
    def on_data_collectors_attached(self) -> None:
        """Set up the data collector when data collectors are attached.

        This method is called automatically by the PAMIQ framework when
        data collectors are attached to the agent.
        """
        super().on_data_collectors_attached()
        self.collector = self.get_data_collector(self.data_collector_name)

    @override
    def step(self, observation: Tensor) -> Tensor:
        """Process an observation and return its encoded representation.

        The method also collects the original observation for potential training.

        Args:
            observation: The input observation tensor (e.g., an image or audio)

        Returns:
            The encoded representation of the observation
        """
        self.collector.collect({DataKey.OBSERVATION: observation.cpu()})
        return self.encoder(observation)
