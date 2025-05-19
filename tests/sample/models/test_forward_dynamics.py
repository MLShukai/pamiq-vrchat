import pytest
import torch
import torch.nn as nn
from pamiq_core.torch import TorchTrainingModel, get_device

from sample.models.components.deterministic_normal import FCDeterministicNormalHead
from sample.models.components.qlstm import QLSTM
from sample.models.forward_dynamics import ForwardDynamics, instantiate
from tests.sample.helpers import parametrize_device


class TestForwardDynamics:
    # Test hyperparameters
    BATCH_SIZE = 2
    SEQ_LEN = 3
    HIDDEN_DEPTH = 2
    HIDDEN_DIM = 8
    OBS_DIM = 16
    ACTION_DIM = 4

    @pytest.fixture
    def observation_flatten(self):
        return nn.Linear(self.OBS_DIM, self.OBS_DIM)

    @pytest.fixture
    def action_flatten(self):
        return nn.Linear(self.ACTION_DIM, self.ACTION_DIM)

    @pytest.fixture
    def obs_action_projection(self):
        combined_dim = self.OBS_DIM + self.ACTION_DIM
        return nn.Linear(combined_dim, self.HIDDEN_DIM)

    @pytest.fixture
    def core_model(self):
        return QLSTM(
            depth=self.HIDDEN_DEPTH,
            dim=self.HIDDEN_DIM,
            dim_ff_hidden=self.HIDDEN_DIM * 2,
            dropout=0.0,
        )

    @pytest.fixture
    def obs_hat_dist_head(self):
        return FCDeterministicNormalHead(self.HIDDEN_DIM, self.OBS_DIM)

    @pytest.fixture
    def dynamics_model(
        self,
        observation_flatten,
        action_flatten,
        obs_action_projection,
        core_model,
        obs_hat_dist_head,
    ):
        return ForwardDynamics(
            observation_flatten,
            action_flatten,
            obs_action_projection,
            core_model,
            obs_hat_dist_head,
        )

    @pytest.fixture
    def obs(self):
        return torch.randn(self.BATCH_SIZE, self.SEQ_LEN, self.OBS_DIM)

    @pytest.fixture
    def action(self):
        return torch.randn(self.BATCH_SIZE, self.SEQ_LEN, self.ACTION_DIM)

    @pytest.fixture
    def hidden(self):
        return torch.randn(self.BATCH_SIZE, self.HIDDEN_DEPTH, self.HIDDEN_DIM)

    def test_forward(self, dynamics_model, obs, action, hidden):
        """Test forward pass of ForwardDynamics model."""
        # Run forward pass
        obs_hat_dist, next_hidden = dynamics_model(obs, action, hidden)

        # Check output types and shapes
        sample = obs_hat_dist.sample()
        assert sample.shape == (self.BATCH_SIZE, self.SEQ_LEN, self.OBS_DIM)
        assert next_hidden.shape == (
            self.BATCH_SIZE,
            self.HIDDEN_DEPTH,
            self.SEQ_LEN,
            self.HIDDEN_DIM,
        )

        # Check distribution properties
        log_prob = obs_hat_dist.log_prob(sample)
        assert log_prob.shape == (self.BATCH_SIZE, self.SEQ_LEN, self.OBS_DIM)

        # Test with different batch sizes
        single_obs = obs[:1]
        single_action = action[:1]
        single_hidden = hidden[:1]

        single_obs_hat_dist, single_next_hidden = dynamics_model(
            single_obs, single_action, single_hidden
        )
        single_sample = single_obs_hat_dist.sample()

        assert single_sample.shape == (1, self.SEQ_LEN, self.OBS_DIM)
        assert single_next_hidden.shape == (
            1,
            self.HIDDEN_DEPTH,
            self.SEQ_LEN,
            self.HIDDEN_DIM,
        )


class TestInstantiate:
    """Tests for the instantiate function in forward_dynamics module."""

    def test_instantiate_creates_valid_model(self):
        """Test that instantiate creates a valid TorchTrainingModel with
        ForwardDynamics."""
        model = instantiate(
            obs_dim=32,
            action_choices=[3, 4],
            action_dim=8,
            depth=2,
            dim=64,
            dim_ff_hidden=128,
            dropout=0.1,
        )

        # Check model type
        assert isinstance(model, TorchTrainingModel)
        assert isinstance(model.model, ForwardDynamics)

    def test_model_forward_pass(self):
        """Test that the instantiated model can perform forward pass with
        correct shapes."""
        model = instantiate(
            obs_dim=32,
            action_choices=[3, 4],
            action_dim=8,
            depth=2,
            dim=64,
            dim_ff_hidden=128,
            dropout=0.1,
        )

        observations = torch.randn(2, 3, 32)
        actions = torch.stack(
            [torch.randint(0, 3, (2, 3)), torch.randint(0, 4, (2, 3))], dim=-1
        )
        hidden = torch.randn(2, 2, 64)

        # Test forward pass
        obs_dist, next_hidden = model.model(observations, actions, hidden)

        # Check output shapes
        assert next_hidden.shape == (2, 2, 3, 64)

        # Check distribution output
        pred_sample = obs_dist.sample()
        assert pred_sample.shape == (2, 3, 32)

        # Check log probability calculation
        log_prob = obs_dist.log_prob(observations)
        assert log_prob.shape == (2, 3, 32)

    @parametrize_device
    def test_instantiate_with_device(self, device):
        """Test that instantiate correctly places the model on specified
        device."""
        model = instantiate(
            obs_dim=32,
            action_choices=[3, 4],
            action_dim=8,
            depth=2,
            dim=64,
            dim_ff_hidden=128,
            dropout=0.1,
            device=device,
        )

        # Check device placement
        assert get_device(model.model) == device

        observations = torch.randn(1, 2, 32, device=device)
        actions = torch.stack(
            [
                torch.randint(0, 3, (1, 2), device=device),
                torch.randint(0, 4, (1, 2), device=device),
            ],
            dim=-1,
        )
        hidden = torch.randn(1, 2, 64, device=device)

        # Should not raise device mismatch errors
        obs_dist, next_hidden = model.model(observations, actions, hidden)

        # Check outputs are on correct device
        assert next_hidden.device == device
        assert obs_dist.sample().device == device
