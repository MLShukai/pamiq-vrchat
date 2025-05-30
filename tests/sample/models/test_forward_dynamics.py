import pytest
import torch
import torch.nn as nn

from sample.models.components.deterministic_normal import FCDeterministicNormalHead
from sample.models.components.qlstm import QLSTM
from sample.models.forward_dynamics import ForwardDynamics


class TestForwardDynamics:
    # Test hyperparameters
    BATCH_SIZE = 2
    SEQ_LEN = 3
    DEPTH = 2
    DIM = 8
    OBS_DIM = 16
    ACTION_DIM = 4
    ACTION_CHOICES = [2, 3, 4]

    @pytest.fixture
    def dynamics_model(self):
        return ForwardDynamics(
            self.OBS_DIM,
            self.ACTION_CHOICES,
            self.ACTION_DIM,
            self.DIM,
            self.DEPTH,
            self.DIM * 2,
        )

    @pytest.fixture
    def obs(self):
        return torch.randn(self.BATCH_SIZE, self.SEQ_LEN, self.OBS_DIM)

    @pytest.fixture
    def action(self):
        actions = []
        for choice in self.ACTION_CHOICES:
            actions.append(torch.randint(0, choice, (self.BATCH_SIZE, self.SEQ_LEN)))
        return torch.stack(actions, dim=-1)

    @pytest.fixture
    def hidden(self):
        return torch.randn(self.BATCH_SIZE, self.DEPTH, self.DIM)

    def test_forward(self, dynamics_model, obs, action, hidden):
        """Test forward pass of ForwardDynamics model."""
        # Run forward pass
        obs_hat_dist, next_hidden = dynamics_model(obs, action, hidden)

        # Check output types and shapes
        sample = obs_hat_dist.sample()
        assert sample.shape == (self.BATCH_SIZE, self.SEQ_LEN, self.OBS_DIM)
        assert next_hidden.shape == (
            self.BATCH_SIZE,
            self.DEPTH,
            self.SEQ_LEN,
            self.DIM,
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
            self.DEPTH,
            self.SEQ_LEN,
            self.DIM,
        )
