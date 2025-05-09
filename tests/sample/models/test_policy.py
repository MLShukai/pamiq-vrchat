import pytest
import torch
import torch.nn as nn

from sample.models.components.fc_scalar_head import FCScalarHead
from sample.models.components.multi_discretes import FCMultiCategoricalHead
from sample.models.components.qlstm import QLSTM
from sample.models.policy import PolicyValueCommon


class TestPolicyValueCommon:
    # Test hyperparameters
    BATCH_SIZE = 2
    SEQ_LEN = 3
    HIDDEN_DEPTH = 2
    HIDDEN_DIM = 8
    OBS_DIM = 16
    ACTION_CATEGORIES = [3, 4, 2]  # Three discrete action dimensions

    @pytest.fixture
    def observation_flatten(self):
        return nn.Linear(self.OBS_DIM, self.HIDDEN_DIM)

    @pytest.fixture
    def core_model(self):
        return QLSTM(
            depth=self.HIDDEN_DEPTH,
            dim=self.HIDDEN_DIM,
            dim_ff_hidden=self.HIDDEN_DIM * 2,
            dropout=0.0,
        )

    @pytest.fixture
    def policy_head(self):
        return FCMultiCategoricalHead(self.HIDDEN_DIM, self.ACTION_CATEGORIES)

    @pytest.fixture
    def value_head(self):
        return FCScalarHead(self.HIDDEN_DIM, squeeze_scalar_dim=True)

    @pytest.fixture
    def policy_value_model(
        self, observation_flatten, core_model, policy_head, value_head
    ):
        return PolicyValueCommon(
            observation_flatten, core_model, policy_head, value_head
        )

    @pytest.fixture
    def observation(self):
        return torch.randn(self.BATCH_SIZE, self.SEQ_LEN, self.OBS_DIM)

    @pytest.fixture
    def hidden(self):
        return torch.randn(self.BATCH_SIZE, self.HIDDEN_DEPTH, self.HIDDEN_DIM)

    def test_forward(self, policy_value_model, observation, hidden):
        """Test forward pass of PolicyValueCommon model."""
        # Run forward pass
        policy_dist, value, next_hidden = policy_value_model(observation, hidden)

        policy_sample = policy_dist.sample()
        assert policy_sample.shape == (
            self.BATCH_SIZE,
            self.SEQ_LEN,
            len(self.ACTION_CATEGORIES),
        )

        assert value.shape == (self.BATCH_SIZE, self.SEQ_LEN)
        assert next_hidden.shape == (
            self.BATCH_SIZE,
            self.HIDDEN_DEPTH,
            self.SEQ_LEN,
            self.HIDDEN_DIM,
        )

        # Test with different batch sizes
        single_obs = observation[:1]
        single_hidden = hidden[:1]

        single_policy_dist, single_value, single_next_hidden = policy_value_model(
            single_obs, single_hidden
        )
        single_policy_sample = single_policy_dist.sample()

        assert single_policy_sample.shape == (
            1,
            self.SEQ_LEN,
            len(self.ACTION_CATEGORIES),
        )
        assert single_value.shape == (1, self.SEQ_LEN)
        assert single_next_hidden.shape == (
            1,
            self.HIDDEN_DEPTH,
            self.SEQ_LEN,
            self.HIDDEN_DIM,
        )

        # Test distribution properties
        log_prob = policy_dist.log_prob(policy_sample)
        assert log_prob.shape == (
            self.BATCH_SIZE,
            self.SEQ_LEN,
            len(self.ACTION_CATEGORIES),
        )
