import pytest
import torch
from pamiq_core.testing import (
    connect_components,
    create_mock_buffer,
    create_mock_models,
)
from pytest_mock import MockerFixture
from tensordict import TensorDict

from sample.agents.temporal_encoding import TemporalEncodingAgent
from sample.data import BufferName, DataKey
from sample.models import ModelName


class TestTemporalEncodingAgent:
    """Tests for the TemporalEncodingAgent class."""

    @pytest.fixture
    def models(self, mocker: MockerFixture):
        training_model, _ = create_mock_models()
        return {ModelName.TEMPORAL_ENCODER: training_model}

    @pytest.fixture
    def buffers(self):
        return {BufferName.TEMPORAL: create_mock_buffer()}

    @pytest.fixture
    def initial_hidden(self):
        return torch.zeros(2, 4, 256)

    def test_initialization(self, models, buffers, initial_hidden):
        """Test initialization of the agent."""
        agent = TemporalEncodingAgent(initial_hidden)

        assert torch.equal(agent.encoder_hidden_state, initial_hidden)

        components = connect_components(agent, buffers=buffers, models=models)

        assert agent.encoder is components.inference_models[ModelName.TEMPORAL_ENCODER]
        assert agent.collector is components.data_collectors[BufferName.TEMPORAL]

    @pytest.mark.parametrize(
        "observation",
        [
            {"image": torch.randn(3, 32, 32), "audio": torch.randn(2, 1600)},
            {"image": torch.randn(1, 3, 64, 64)},
        ],
    )
    def test_step(self, observation, models, buffers, mocker: MockerFixture):
        """Test that the agent correctly encodes temporal observations and
        maintains hidden state."""
        initial_hidden = torch.randn(2, 4, 256)
        agent = TemporalEncodingAgent(initial_hidden)

        components = connect_components(agent, buffers=buffers, models=models)

        output_tensor = torch.ones(32, 256)
        next_hidden = torch.randn_like(initial_hidden)
        components.inference_models[ModelName.TEMPORAL_ENCODER].return_value = (  # pyright: ignore[reportAttributeAccessIssue, ]
            output_tensor,
            next_hidden,
        )

        spy_collect = mocker.spy(
            components.data_collectors[BufferName.TEMPORAL], "collect"
        )

        output = agent.step(observation)

        # Verify encoder was called with TensorDict
        agent.encoder.assert_called_once()  # pyright: ignore[reportAttributeAccessIssue, ]
        call_args = agent.encoder.call_args[0]  # pyright: ignore[reportAttributeAccessIssue, ]
        assert isinstance(call_args[0], TensorDict)
        assert torch.equal(call_args[1], initial_hidden)

        # Verify output
        assert torch.equal(output, output_tensor)

        # Verify hidden state was updated
        assert torch.equal(agent.encoder_hidden_state, next_hidden)

        # Verify data collection
        spy_collect.assert_called_once()
        collected_data = spy_collect.call_args[0][0]
        assert DataKey.OBSERVATION in collected_data
        assert DataKey.HIDDEN in collected_data
        assert isinstance(collected_data[DataKey.OBSERVATION], TensorDict)
        assert torch.equal(collected_data[DataKey.HIDDEN], initial_hidden)

    def test_save_and_load_state(self, tmp_path):
        """Test saving and loading the agent state."""
        hidden_state = torch.randn(2, 4, 256)
        agent = TemporalEncodingAgent(hidden_state)

        # Save state
        save_path = tmp_path / "agent_state"
        agent.save_state(save_path)

        # Verify file exists
        assert (save_path / "encoder_hidden_state.pt").is_file()

        # Create new agent with different initial hidden state
        new_agent = TemporalEncodingAgent(torch.zeros_like(hidden_state))

        # Verify hidden state is difference
        assert not torch.equal(new_agent.encoder_hidden_state, hidden_state)

        # Load state
        new_agent.load_state(save_path)

        # Verify hidden state was loaded correctly
        assert torch.equal(new_agent.encoder_hidden_state, hidden_state)
