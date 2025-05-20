import pytest
import torch

from sample.models.temporal_encoder import ObsInfo, TemporalEncoder


class TestTemporalEncoder:
    # Test hyperparameters
    BATCH_SIZE = 2
    SEQ_LEN = 3
    DIM = 8
    DEPTH = 2
    OBS_INFOS = {
        "image": ObsInfo(dim=32, dim_hidden=16, num_tokens=4),
        "audio": ObsInfo(dim=24, dim_hidden=12, num_tokens=3),
    }

    @pytest.fixture
    def encoder(self):
        return TemporalEncoder(self.OBS_INFOS, self.DIM, self.DEPTH, self.DIM * 2, 0.1)

    @pytest.fixture
    def observations(self):
        return {
            k: torch.randn(self.BATCH_SIZE, self.SEQ_LEN, v.num_tokens, v.dim)
            for k, v in self.OBS_INFOS.items()
        }

    @pytest.fixture
    def hidden(self):
        return torch.randn(self.BATCH_SIZE, self.DEPTH, self.DIM)

    def test_forward(self, encoder, observations, hidden):
        """Test forward pass of TemporalEncoder."""
        obs_hat_dists, next_hidden = encoder(observations, hidden)

        # Check output shapes
        assert next_hidden.shape == (
            self.BATCH_SIZE,
            self.DEPTH,
            self.SEQ_LEN,
            self.DIM,
        )

        # Check distribution outputs
        for key, info in self.OBS_INFOS.items():
            assert key in obs_hat_dists
            sample = obs_hat_dists[key].sample()
            assert sample.shape == (
                self.BATCH_SIZE,
                self.SEQ_LEN,
                info.num_tokens,
                info.dim,
            )

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
        assert x.shape == (self.BATCH_SIZE, self.SEQ_LEN, self.DIM)
        assert next_hidden.shape == (
            self.BATCH_SIZE,
            self.DEPTH,
            self.SEQ_LEN,
            self.DIM,
        )

        # Verify layer normalization was applied
        assert torch.abs(x.mean()).item() < 1e-5
        assert torch.abs(x.std() - 1.0).item() < 1e-1
