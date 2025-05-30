from collections.abc import Mapping

import pytest
import torch
from pytest_mock import MockerFixture

from sample.agents.curiosity import CuriosityAgent
from sample.agents.integration import IntegratedCuriosityFramework
from sample.agents.temporal_encoding import TemporalEncodingAgent
from sample.agents.unimodal_encoding import UnimodalEncodingAgent


class TestIntegratedCuriosityFramework:
    """Tests for the IntegratedCuriosityFramework class."""

    @pytest.fixture
    def mock_unimodal_agents(
        self, mocker: MockerFixture
    ) -> Mapping[str, UnimodalEncodingAgent]:
        """Create mock unimodal encoding agents."""
        image_agent = mocker.Mock(spec=UnimodalEncodingAgent)
        image_agent.step.return_value = torch.ones(16, 256)

        audio_agent = mocker.Mock(spec=UnimodalEncodingAgent)
        audio_agent.step.return_value = torch.ones(8, 128)

        return {"image": image_agent, "audio": audio_agent}

    @pytest.fixture
    def mock_temporal_agent(self, mocker: MockerFixture) -> TemporalEncodingAgent:
        """Create mock temporal encoding agent."""
        temporal_agent = mocker.Mock(spec=TemporalEncodingAgent)
        temporal_agent.step.return_value = torch.ones(512)
        return temporal_agent

    @pytest.fixture
    def mock_curiosity_agent(self, mocker: MockerFixture) -> CuriosityAgent:
        """Create mock curiosity agent."""
        curiosity_agent = mocker.Mock(spec=CuriosityAgent)
        curiosity_agent.step.return_value = torch.ones(4)
        return curiosity_agent

    @pytest.fixture
    def framework(
        self, mock_unimodal_agents, mock_temporal_agent, mock_curiosity_agent
    ) -> IntegratedCuriosityFramework:
        """Create the integrated framework with mock agents."""
        return IntegratedCuriosityFramework(
            unimodal_agents=mock_unimodal_agents,
            temporal_agent=mock_temporal_agent,
            curiosity_agent=mock_curiosity_agent,
        )

    def test_initialization(
        self, framework, mock_unimodal_agents, mock_temporal_agent, mock_curiosity_agent
    ):
        """Test successful initialization of the framework."""
        assert framework.unimodals == mock_unimodal_agents
        assert framework.temporal == mock_temporal_agent
        assert framework.curiosity == mock_curiosity_agent

    def test_initialization_with_reserved_names(
        self, mock_temporal_agent, mock_curiosity_agent
    ):
        """Test initialization fails with reserved names."""
        bad_unimodal_agents = {
            "curiosity": mock_curiosity_agent  # Using reserved name
        }

        with pytest.raises(ValueError, match="must not be 'curiosity' or 'temporal'"):
            IntegratedCuriosityFramework(
                unimodal_agents=bad_unimodal_agents,
                temporal_agent=mock_temporal_agent,
                curiosity_agent=mock_curiosity_agent,
            )

    def test_step(
        self, framework, mock_unimodal_agents, mock_temporal_agent, mock_curiosity_agent
    ):
        """Test step method correctly processes observations through all
        agents."""
        # Create test observations
        observations = {"image": torch.randn(3, 224, 224), "audio": torch.randn(16000)}

        # Call step
        action = framework.step(observations)

        # Verify each unimodal agent was called with correct observation
        mock_unimodal_agents["image"].step.assert_called_once()
        mock_unimodal_agents["audio"].step.assert_called_once()
        assert torch.equal(
            mock_unimodal_agents["image"].step.call_args[0][0], observations["image"]
        )
        assert torch.equal(
            mock_unimodal_agents["audio"].step.call_args[0][0], observations["audio"]
        )

        # Verify temporal agent was called with encoded observations
        mock_temporal_agent.step.assert_called_once()
        encoded_obs_arg = mock_temporal_agent.step.call_args[0][0]
        assert "image" in encoded_obs_arg
        assert "audio" in encoded_obs_arg

        # Verify curiosity agent was called with temporal encoding
        mock_curiosity_agent.step.assert_called_once_with(
            mock_temporal_agent.step.return_value
        )

        # Verify final output
        assert torch.equal(action, mock_curiosity_agent.step.return_value)
