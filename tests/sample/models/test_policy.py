import pytest
import torch
import torch.nn as nn
from pamiq_core.torch import TorchTrainingModel, get_device

from sample.models.components.fc_scalar_head import FCScalarHead
from sample.models.components.multi_discretes import FCMultiCategoricalHead
from sample.models.components.qlstm import QLSTM
from sample.models.policy import PolicyValueCommon, instantiate
from tests.sample.helpers import parametrize_device


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


class TestPolicyInstantiate:
    """Tests for the instantiate function in policy module."""

    @pytest.mark.parametrize("obs_dim", [32, 64])
    @pytest.mark.parametrize("action_choices", [[3, 4], [2, 3, 4]])
    def test_instantiate_creates_valid_model(self, obs_dim, action_choices):
        """Test that instantiate creates a valid TorchTrainingModel with
        PolicyValueCommon."""
        model = instantiate(
            obs_dim=obs_dim,
            depth=2,
            dim=64,
            dim_ff_hidden=128,
            dropout=0.1,
            action_choices=action_choices,
        )

        # Check model type
        assert isinstance(model, TorchTrainingModel)
        assert isinstance(model.model, PolicyValueCommon)

        # Check components
        policy_model = model.model
        # Test identity optimizations work correctly
        if obs_dim == 64:  # dim matches obs_dim
            assert isinstance(policy_model.observation_flatten, nn.Identity)
        else:
            assert isinstance(policy_model.observation_flatten, nn.Linear)

    def test_model_forward_pass(self):
        """Test that the instantiated model can perform forward pass with
        correct shapes."""
        obs_dim = 32
        action_choices = [3, 4]
        batch_size = 2
        seq_len = 3

        model = instantiate(
            obs_dim=obs_dim,
            depth=2,
            dim=64,
            dim_ff_hidden=128,
            dropout=0.1,
            action_choices=action_choices,
        )

        # Create sample inputs
        observations = torch.randn(batch_size, seq_len, obs_dim)
        hidden = torch.randn(batch_size, 2, 64)  # batch_size, depth, dim

        # Test forward pass
        policy_dist, value, next_hidden = model.model(observations, hidden)

        # Check output shapes
        assert value.shape == (batch_size, seq_len)
        assert next_hidden.shape == (batch_size, 2, seq_len, 64)

        # Check policy distribution
        action_sample = policy_dist.sample()
        assert action_sample.shape == (batch_size, seq_len, len(action_choices))

        # Check we can get log probabilities
        log_probs = policy_dist.log_prob(action_sample)
        assert log_probs.shape == (batch_size, seq_len, len(action_choices))

    @parametrize_device
    def test_instantiate_with_device(self, device):
        """Test that instantiate correctly places the model on specified
        device."""
        model = instantiate(
            obs_dim=32,
            depth=2,
            dim=64,
            dim_ff_hidden=128,
            dropout=0.1,
            action_choices=[3, 4],
            device=device,
        )

        # Check device placement
        assert get_device(model.model) == device

        # Test with some input on the correct device
        observations = torch.randn(1, 3, 32, device=device)
        hidden = torch.randn(1, 2, 64, device=device)

        # Should not raise device mismatch errors
        policy_dist, value, next_hidden = model.model(observations, hidden)

        # Check outputs are on correct device
        assert value.device == device
        assert next_hidden.device == device
        assert policy_dist.sample().device == device
