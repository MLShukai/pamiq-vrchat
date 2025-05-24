import re

import numpy as np
import pytest
import torch

from sample.transforms.audio import (
    AudioFrameToTensor,
    AudioLengthCompletion,
    create_transform,
)


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


class TestAudioLengthCompletion:
    """Tests for the AudioLengthCompletion class."""

    @pytest.mark.parametrize("invalid_frame_size", [0, -1, -10])
    def test_initialization_invalid_frame_size(self, invalid_frame_size: int):
        """Test that initialization fails with invalid frame sizes."""
        with pytest.raises(ValueError, match="frame_size must be positive"):
            AudioLengthCompletion(invalid_frame_size)

    @pytest.mark.parametrize(
        "frame_size,n_channels,n_audios,audio_length",
        [
            (10, 2, 5, 7),  # Completion required: audio shorter than frame
            (10, 1, 3, 8),  # Completion required: mono audio
            (50, 2, 10, 30),  # Completion required: longer sequence
        ],
    )
    def test_completion_required(
        self,
        frame_size: int,
        n_channels: int,
        n_audios: int,
        audio_length: int,
    ):
        """Test audio completion when input is shorter than frame size.

        This test verifies that the module correctly accumulates short
        audio chunks into a buffer and outputs the required frame size.
        """
        assert audio_length < frame_size, "Test expects audio shorter than frame size"

        completion = AudioLengthCompletion(frame_size)

        # Generate sequence of audio chunks
        audio_chunks = [torch.randn(n_channels, audio_length) for _ in range(n_audios)]

        # Concatenate all chunks to compute expected final state
        all_audio = torch.cat(audio_chunks, dim=-1)

        # Process each chunk and verify output shape
        for i, audio_chunk in enumerate(audio_chunks):
            output = completion(audio_chunk)

            # Output should always have the target frame size
            assert output.shape == (n_channels, frame_size)

            # Verify the content matches expected buffer state
            end_index = audio_length * (i + 1)
            start_index = max(0, end_index - frame_size)
            expected_content = all_audio[:, start_index:end_index]

            # Add zero padding if needed
            if expected_content.size(-1) < frame_size:
                padding_size = frame_size - expected_content.size(-1)
                zero_pad = torch.zeros(n_channels, padding_size)
                expected_content = torch.cat([zero_pad, expected_content], dim=-1)

            # Compare actual output with expected content
            assert torch.allclose(output, expected_content, atol=1e-6)

    @pytest.mark.parametrize("frame_size,n_channels", [(10, 2), (20, 1), (50, 4)])
    def test_no_completion_needed(self, frame_size: int, n_channels: int):
        """Test behavior when input audio is exactly the frame size.

        When input length equals frame size, the module should return
        the input unchanged without buffering.
        """
        completion = AudioLengthCompletion(frame_size)

        # Create audio with exact frame size
        audio = torch.randn(n_channels, frame_size)
        output = completion(audio)

        # Output should be identical to input
        assert output.shape == (n_channels, frame_size)
        assert torch.equal(output, audio)

        # Buffer should remain None since no accumulation was needed
        assert completion.buffer is None

    @pytest.mark.parametrize(
        "frame_size,n_channels,audio_length",
        [
            (10, 2, 15),  # Audio longer than frame
            (20, 1, 30),  # Much longer audio
            (50, 3, 100),  # Very long audio
        ],
    )
    def test_cutting_required(
        self,
        frame_size: int,
        n_channels: int,
        audio_length: int,
    ):
        """Test audio cutting when input is longer than frame size.

        The module should return only the last frame_size samples when
        input is longer than required.
        """
        assert audio_length > frame_size, "Test expects audio longer than frame size"

        completion = AudioLengthCompletion(frame_size)

        # Create audio longer than frame size
        audio = torch.randn(n_channels, audio_length)
        output = completion(audio)

        # Output should be the last frame_size samples
        expected_output = audio[:, -frame_size:]
        assert output.shape == (n_channels, frame_size)
        assert torch.equal(output, expected_output)

        # Buffer should remain None since no accumulation was needed
        assert completion.buffer is None

    def test_incompatible_buffer_shape_error(self):
        """Test error handling for incompatible input shapes.

        If the buffer exists and new input has incompatible dimensions,
        a clear error should be raised.
        """
        completion = AudioLengthCompletion(10)

        # First input with stereo audio
        stereo_audio = torch.randn(2, 5)
        completion(stereo_audio)

        # Try to process mono audio (incompatible shape)
        mono_audio = torch.randn(1, 5)

        with pytest.raises(
            ValueError,
            match=re.escape(
                "Input shape (1, 5) is incompatible with buffer shape (2, 10). "
                "All dimensions except the last must match."
            ),
        ):
            completion(mono_audio)

    def test_buffer_shape_compatibility_different_lengths(self):
        """Test that different audio lengths are compatible with same channel
        count.

        As long as all dimensions except the last match, different audio
        lengths should be processed correctly.
        """
        completion = AudioLengthCompletion(15)

        # Process audio of different lengths but same channel count
        audio1 = torch.randn(2, 5)
        output1 = completion(audio1)
        assert output1.shape == (2, 15)

        audio2 = torch.randn(2, 8)  # Different length, same channels
        output2 = completion(audio2)
        assert output2.shape == (2, 15)

        # Should work without errors
        audio3 = torch.randn(2, 3)
        output3 = completion(audio3)
        assert output3.shape == (2, 15)


class TestCreateTransform:
    @pytest.mark.parametrize(
        "source_rate,target_rate,frame_size,target_size,channels",
        [(16000, 8000, 1600, 3200, 1), (44100, 16000, 1024, 128, 2)],
    )
    def test_full_pipeline(
        self, source_rate, target_rate, frame_size, target_size, channels
    ):
        """Test the full audio transform pipeline."""
        transform = create_transform(source_rate, target_rate, target_size)
        audio_frame = np.random.randn(frame_size, channels).astype(np.float32)

        output = transform(audio_frame)

        # Check output properties
        assert isinstance(output, torch.Tensor)
        # Expected output length after resampling
        assert output.shape == (channels, target_size)

        # Check standardization
        assert torch.abs(output.mean()) < 1e-5
        assert torch.abs(output.std() - 1.0) < 1e-1  # Allow some tolerance
