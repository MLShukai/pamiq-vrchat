import pytest
import torch
import torch.nn as nn
from pamiq_core.torch import TorchTrainingModel, get_device

from sample.models.components.deterministic_normal import FCDeterministicNormalHead
from sample.models.components.qlstm import QLSTM
from sample.models.temporal_encoder import ObsInfo, TemporalEncoder, instantiate
from tests.sample.helpers import parametrize_device


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
        obs_hat_dists, next_hidden = encoder(observations, hidden)

        # Check output shapes
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


class TestInstantiate:
    """Tests for the instantiate function in temporal_encoder module."""

    @pytest.fixture
    def obs_infos(self):
        """Create sample observation configurations."""
        return {
            "image": ObsInfo(dim_in=32, dim_out=16, num_tokens=4),
            "audio": ObsInfo(dim_in=24, dim_out=12, num_tokens=3),
        }

    def test_instantiate_creates_valid_model(self, obs_infos):
        """Test that instantiate creates a valid TorchTrainingModel with
        TemporalEncoder."""
        model = instantiate(
            obs_infos=obs_infos, depth=2, dim=64, dim_ff_hidden=128, dropout=0.1
        )

        # Check model type
        assert isinstance(model, TorchTrainingModel)
        assert isinstance(model.model, TemporalEncoder)

    @parametrize_device
    def test_instantiate_with_device(self, device, obs_infos):
        """Test that instantiate correctly places the model on specified
        device."""
        model = instantiate(
            obs_infos=obs_infos,
            depth=2,
            dim=64,
            dim_ff_hidden=128,
            dropout=0.1,
            device=device,
        )

        assert get_device(model.model) == device

        # Test with some input on the correct device
        observations = {
            "image": torch.randn(1, 4, 32, device=device),
            "audio": torch.randn(1, 3, 24, device=device),
        }

        hidden = torch.randn(1, 2, 64, device=device)

        # Should not raise device mismatch errors
        obs_hat_dists, next_hidden = model.model(observations, hidden)
        assert obs_hat_dists["image"].sample().shape == (1, 4, 32)
        assert obs_hat_dists["audio"].sample().shape == (1, 3, 24)
        assert next_hidden.device == device

    def test_instantiate_model_inference(self, obs_infos):
        """Test that the instantiated model can perform inference with correct
        shapes."""
        model = instantiate(
            obs_infos=obs_infos, depth=2, dim=64, dim_ff_hidden=128, dropout=0.1
        )

        # Create sample observations
        observations = {
            "image": torch.randn(4, 32),  # tokens=4, dim=32
            "audio": torch.randn(3, 24),  # tokens=3, dim=24
        }

        hidden = torch.randn(2, 64)  # depth=2, dim=64

        # Test inference
        x, next_hidden = model.inference_model(observations, hidden)

        # Check output shapes
        assert x.shape == (64,)  # encoded feature
        assert next_hidden.shape == (2, 64)  # batch_size, depth, seq_len, dim
