import pytest
import torch
import torch.nn as nn

from sample.models.components.deterministic_normal import FCDeterministicNormalHead
from sample.models.components.qlstm import QLSTM
from sample.models.temporal_encoder import TemporalEncoder


class TestTemporalEncoder:
    # Test hyperparameters
    BATCH_SIZE = 2
    SEQ_LEN = 3
    HIDDEN_DEPTH = 2
    HIDDEN_DIM = 8
    MODAL_DIMS = {"image": 16, "audio": 12}

    @pytest.fixture
    def observation_flattens(self):
        return {k: nn.Linear(dim, dim) for k, dim in self.MODAL_DIMS.items()}

    @pytest.fixture
    def flattened_obses_projection(self):
        total_dim = sum(self.MODAL_DIMS.values())
        return nn.Linear(total_dim, self.HIDDEN_DIM)

    @pytest.fixture
    def core_model(self):
        return QLSTM(
            depth=self.HIDDEN_DEPTH,
            dim=self.HIDDEN_DIM,
            dim_ff_hidden=self.HIDDEN_DIM * 2,
            dropout=0.0,
        )

    @pytest.fixture
    def obs_hat_dist_heads(self):
        return {
            k: FCDeterministicNormalHead(self.HIDDEN_DIM, dim)
            for k, dim in self.MODAL_DIMS.items()
        }

    @pytest.fixture
    def encoder(
        self,
        observation_flattens,
        flattened_obses_projection,
        core_model,
        obs_hat_dist_heads,
    ):
        return TemporalEncoder(
            observation_flattens,
            flattened_obses_projection,
            core_model,
            obs_hat_dist_heads,
        )

    @pytest.fixture
    def observations(self):
        return {
            k: torch.randn(self.BATCH_SIZE, self.SEQ_LEN, dim)
            for k, dim in self.MODAL_DIMS.items()
        }

    @pytest.fixture
    def hidden(self):
        return torch.randn(self.BATCH_SIZE, self.HIDDEN_DEPTH, self.HIDDEN_DIM)

    def test_init_key_mismatch(
        self,
        observation_flattens,
        flattened_obses_projection,
        core_model,
        obs_hat_dist_heads,
    ):
        """Test initialization with mismatched keys raises KeyError."""
        mismatched_flattens = observation_flattens.copy()
        mismatched_flattens["text"] = nn.Linear(10, 10)

        with pytest.raises(KeyError, match="not matched"):
            TemporalEncoder(
                mismatched_flattens,
                flattened_obses_projection,
                core_model,
                obs_hat_dist_heads,
            )

    def test_forward(self, encoder, observations, hidden):
        """Test forward pass of TemporalEncoder."""
        x, next_hidden, obs_hat_dists = encoder(observations, hidden)

        # Check output shapes
        assert x.shape == (self.BATCH_SIZE, self.SEQ_LEN, self.HIDDEN_DIM)
        assert next_hidden.shape == (
            self.BATCH_SIZE,
            self.HIDDEN_DEPTH,
            self.SEQ_LEN,
            self.HIDDEN_DIM,
        )

        # Check distribution outputs
        for key, modal_dim in self.MODAL_DIMS.items():
            assert key in obs_hat_dists
            sample = obs_hat_dists[key].sample()
            assert sample.shape == (self.BATCH_SIZE, self.SEQ_LEN, modal_dim)

    def test_forward_key_mismatch(self, encoder, observations, hidden):
        """Test forward pass with mismatched observation keys raises
        KeyError."""
        invalid_observations = {"image": observations["image"]}

        with pytest.raises(KeyError, match="not matched"):
            encoder(invalid_observations, hidden)

    def test_infer(self, encoder, observations, hidden):
        """Test infer method of TemporalEncoder."""
        x, next_hidden = encoder.infer(observations, hidden)

        # Check output shapes
        assert x.shape == (self.BATCH_SIZE, self.SEQ_LEN, self.HIDDEN_DIM)
        assert next_hidden.shape == (
            self.BATCH_SIZE,
            self.HIDDEN_DEPTH,
            self.SEQ_LEN,
            self.HIDDEN_DIM,
        )

        # Verify layer normalization was applied
        assert torch.abs(x.mean()).item() < 1e-5
        assert torch.abs(x.std() - 1.0).item() < 1e-1
