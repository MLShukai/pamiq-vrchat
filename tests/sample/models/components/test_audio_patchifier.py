import pytest
import torch

from sample.models.components.audio_patchifier import AudioPatchifier


class TestAudioPatchifier:
    @pytest.mark.parametrize("batch_size", [1, 4])
    @pytest.mark.parametrize("sample_size", [400, 16080])
    @pytest.mark.parametrize(
        "in_channels", [1, 2]
    )  # monoral and stereo audio respectively.
    @pytest.mark.parametrize("embed_dim", [512])
    def test_forward(
        self, batch_size: int, sample_size: int, in_channels: int, embed_dim: int
    ):
        audio_patchifier = AudioPatchifier(in_channels=in_channels, embed_dim=embed_dim)
        input_audios = torch.randn(batch_size, in_channels, sample_size)
        output_patches = audio_patchifier(input_audios)
        assert isinstance(output_patches, torch.Tensor)
        # According to data2vec paper (https://arxiv.org/abs/2202.03555),
        # assuming window_size=400[samples] and hop_size=320[samples] as total.
        window_size, hop_size = 400, 320
        expected_n_patches = int((sample_size - window_size) / hop_size) + 1
        assert output_patches.shape == (batch_size, expected_n_patches, embed_dim)

    @pytest.mark.parametrize(
        "sample_size,batch_size,in_channels,embed_dim",
        [
            (400, 1, 2, 512),  # Minimum case
            (16080, 4, 2, 512),  # Standard case
            (32000, 2, 1, 768),  # Larger case
            (800, 3, 1, 256),  # Small case with different params
        ],
    )
    def test_compute_num_patches_consistency_with_forward(
        self, sample_size: int, batch_size: int, in_channels: int, embed_dim: int
    ):
        """Test that compute_num_patches matches actual forward pass output."""
        # Compute expected number of patches
        expected_patches = AudioPatchifier.compute_num_patches(sample_size)

        # Create patchifier and run forward pass
        audio_patchifier = AudioPatchifier(in_channels=in_channels, embed_dim=embed_dim)
        input_audio = torch.randn(batch_size, in_channels, sample_size)
        output = audio_patchifier(input_audio)

        # Check consistency
        assert output.shape[1] == expected_patches
        assert output.shape == (batch_size, expected_patches, embed_dim)
