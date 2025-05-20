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
