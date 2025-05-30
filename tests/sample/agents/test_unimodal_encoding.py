import pytest
import torch
from pamiq_core import DataBuffer, InferenceModel, TrainingModel
from pamiq_core.testing import connect_components
from pytest_mock import MockerFixture

from sample.agents.unimodal_encoding import UnimodalEncodingAgent
from sample.data import DataKey


class TestUnimodalEncodingAgent:
    """Tests for the UnimodalEncodingAgent class."""

    @pytest.fixture
    def models(self, mocker: MockerFixture):
        training_model = mocker.Mock(TrainingModel)
        training_model.inference_model = mocker.Mock(InferenceModel)
        training_model.has_inference_model = True

        return {"encoder": training_model}

    @pytest.fixture
    def buffers(self, mocker: MockerFixture):
        buf = mocker.MagicMock(DataBuffer)
        buf.max_size = 0
        return {"data": buf}

    def test_initilization(self, models, buffers):
        """Test initialization of the agent."""
        model_name = "encoder"
        data_collector_name = "data"

        agent = UnimodalEncodingAgent(model_name, data_collector_name)

        assert agent.model_name == model_name
        assert agent.data_collector_name == data_collector_name

        components = connect_components(agent, buffers=buffers, models=models)

        assert agent.encoder is components.inference_models["encoder"]
        assert agent.collector is components.data_collectors["data"]

    @pytest.mark.parametrize(
        "input_shape,expected_output_shape",
        [
            ((3, 32, 32), (16, 1, 1)),
            ((2, 1600), (8, 1)),
        ],
    )
    def test_step(
        self, input_shape, expected_output_shape, models, buffers, mocker: MockerFixture
    ):
        """Test that the agent correctly encodes observations and collects
        data."""
        agent = UnimodalEncodingAgent("encoder", "data")

        components = connect_components(agent, buffers=buffers, models=models)
        components.inference_models["encoder"].return_value = torch.ones(  # pyright: ignore[reportAttributeAccessIssue]
            expected_output_shape
        )

        spy_collect = mocker.spy(components.data_collectors["data"], "collect")

        observation = torch.randn(input_shape)
        output = agent.step(observation)

        agent.encoder.assert_called_once_with(observation)  # pyright: ignore[reportAttributeAccessIssue]
        assert output.shape == expected_output_shape

        spy_collect.assert_called_once()
        call_args = spy_collect.call_args[0][0]
        assert DataKey.OBSERVATION in call_args
        assert torch.equal(call_args[DataKey.OBSERVATION], observation)
