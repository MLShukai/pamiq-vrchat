from collections.abc import Mapping
from typing import override

from pamiq_core import Agent
from torch import Tensor

from .curiosity import CuriosityAgent
from .temporal_encoding import TemporalEncodingAgent
from .unimodal_encoding import UnimodalEncodingAgent


class IntegratedCuriosityFramework(Agent[Mapping[str, Tensor], Tensor]):
    """Integrated framework combining unimodal encoding, temporal encoding, and
    curiosity-driven exploration.

    This agent integrates three types of subagents to create a complete perception-action
    loop for a curiosity-driven learning system:

    1. Unimodal Encoding Agents: Process individual modality inputs (e.g., vision, audio)
    2. Temporal Encoding Agent: Combines encoded modalities and extracts temporal features
    3. Curiosity Agent: Selects actions based on temporal features and intrinsic motivation

    The framework processes observations through each layer sequentially, first encoding
    each modality independently, then integrating them temporally, and finally selecting
    actions based on curiosity-driven exploration.
    """

    def __init__(
        self,
        unimodal_agents: Mapping[str, UnimodalEncodingAgent],
        temporal_agent: TemporalEncodingAgent,
        curiosity_agent: CuriosityAgent,
    ) -> None:
        """Initialize the integrated curiosity framework.

        Args:
            unimodal_agents: Dictionary mapping modality names to unimodal encoding agents
            temporal_agent: Agent that processes multimodal encodings with temporal context
            curiosity_agent: Agent that selects actions based on intrinsic motivation

        Raises:
            ValueError: If any unimodal agent is named 'curiosity' or 'temporal'
        """
        if "curiosity" in unimodal_agents or "temporal" in unimodal_agents:
            raise ValueError(
                "Unimodal agent name must not be 'curiosity' or 'temporal'."
            )
        super().__init__(
            agents={
                **unimodal_agents,
                "temporal": temporal_agent,
                "curiosity": curiosity_agent,
            }
        )

        self.unimodals = unimodal_agents
        self.temporal = temporal_agent
        self.curiosity = curiosity_agent

    @override
    def step(self, observation: Mapping[str, Tensor]) -> Tensor:
        """Process multimodal observations and produce actions.

        The processing follows these steps:
        1. Encode each modality observation with corresponding unimodal agent
        2. Integrate encoded observations via temporal encoding agent
        3. Select action using curiosity-driven agent

        Args:
            observation: Dictionary mapping modality names to observation tensors

        Returns:
            Action tensor selected by the curiosity agent
        """
        encoded_obs = {k: self.unimodals[k].step(obs) for k, obs in observation.items()}

        temporal_encoded = self.temporal.step(encoded_obs)

        action = self.curiosity.step(temporal_encoded)

        return action
