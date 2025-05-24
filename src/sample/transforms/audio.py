from collections.abc import Callable
from typing import override

import torch
import torch.nn as nn
from torchaudio.transforms import Resample

from pamiq_vrchat.sensors import AudioFrame

from .common import Standardize


class AudioFrameToTensor(nn.Module):
    """Converts AudioFrame (numpy array) to PyTorch tensor.

    This module transforms audio frames from NumPy array format to
    PyTorch tensors and transposes the dimensions for compatibility with
    torchaudio transforms.
    """

    @override
    def forward(self, frame: AudioFrame) -> torch.Tensor:
        """Convert audio frame to tensor.

        Args:
            frame: Input audio frame as a NumPy array with shape (frame_size, channels).

        Returns:
            Audio tensor with shape (channels, frame_size).
        """
        return torch.from_numpy(frame).transpose(0, 1)


class LengthCompletion(nn.Module):
    """Complete audio length using previously buffered audio data.

    This module maintains an internal buffer to handle cases where input
    audio sequences are shorter than the required frame size. It
    concatenates new input with the buffered audio to ensure output
    always has the target length.
    """

    def __init__(self, frame_size: int) -> None:
        """Initialize the length completion module.

        Args:
            frame_size: Target number of samples in the output audio.

        Raises:
            ValueError: If frame_size is not positive.
        """
        super().__init__()
        if frame_size <= 0:
            raise ValueError("frame_size must be positive")

        self.frame_size = frame_size
        # Register buffer as None initially - will be created on first forward pass
        self.register_buffer("buffer", None)

    @override
    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Process audio data and ensure output has the target frame size.

        Args:
            data: Input audio tensor with shape (..., input_length).

        Returns:
            Audio tensor with adjusted length and shape (..., frame_size).

        Raises:
            ValueError: If input tensor has incompatible shape with existing buffer.
        """
        # Validate buffer compatibility if it exists
        if self.buffer is not None:
            if (
                self.buffer.ndim != data.ndim
                or self.buffer.shape[:-1] != data.shape[:-1]
            ):
                raise ValueError(
                    f"Input shape {tuple(data.shape)} is incompatible with buffer shape "
                    f"{tuple(self.buffer.shape)}. All dimensions except the last must match."
                )

        # If input is already long enough, return the last frame_size samples
        if data.size(-1) >= self.frame_size:
            return data[..., -self.frame_size :]

        # Initialize buffer if this is the first call
        if self.buffer is None:
            self.buffer = torch.zeros(
                (*data.shape[:-1], self.frame_size),
                dtype=data.dtype,
                device=data.device,
            )

        # Concatenate buffer with new data and keep only the last frame_size samples
        self.buffer = torch.cat([self.buffer, data], dim=-1)[..., -self.frame_size :]

        # Return a copy to prevent external modifications to the buffer
        return self.buffer.clone()


def create_vrchat_transform(
    source_sample_rate: int,
    target_sample_rate: int,
) -> Callable[[AudioFrame], torch.Tensor]:
    """Create a composed transform for VRChat audio processing.

    Creates a transform pipeline that:
    1. Converts audio frame to tensor
    2. Resamples to target sample rate
    3. Standardizes values (zero mean, unit variance)

    Args:
        source_sample_rate: Original sample rate of audio in Hz.
        target_sample_rate: Target sample rate to resample to in Hz.

    Returns:
        A callable that transforms AudioFrame to standardized tensor.
    """
    return nn.Sequential(
        AudioFrameToTensor(),
        Resample(source_sample_rate, target_sample_rate),
        Standardize(),
    )
