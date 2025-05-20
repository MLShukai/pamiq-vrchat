import numpy as np
import pytest
import torch

from sample.transforms.audio import AudioFrameToTensor, create_vrchat_transform


class TestAudioFrameToTensor:
    @pytest.mark.parametrize(
        "frame_size,channels",
        [(1024, 1), (512, 2), (2048, 4)],
    )
    def test_conversion(self, frame_size, channels):
        """Test conversion from numpy array to tensor with correct shape."""
        transform = AudioFrameToTensor()
        audio_frame = np.random.randn(frame_size, channels).astype(np.float32)

        output = transform(audio_frame)

        assert isinstance(output, torch.Tensor)
        assert output.shape == (channels, frame_size)
        assert output.dtype == torch.float32

        # Check transposition was done correctly
        np_transposed = audio_frame.transpose(1, 0)
        assert torch.allclose(output, torch.from_numpy(np_transposed))


class TestCreateVRChatTransform:
    @pytest.mark.parametrize(
        "source_rate,target_rate,frame_size,channels",
        [(16000, 8000, 1600, 1), (44100, 16000, 1024, 2)],
    )
    def test_full_pipeline(self, source_rate, target_rate, frame_size, channels):
        """Test the full audio transform pipeline."""
        transform = create_vrchat_transform(source_rate, target_rate)
        audio_frame = np.random.randn(frame_size, channels).astype(np.float32)

        output = transform(audio_frame)

        # Check output properties
        assert isinstance(output, torch.Tensor)
        # Expected output length after resampling
        expected_length = round(frame_size * target_rate / source_rate)
        assert output.shape == (channels, expected_length)

        # Check standardization
        assert torch.abs(output.mean()) < 1e-5
        assert torch.abs(output.std() - 1.0) < 1e-1  # Allow some tolerance
