"""Tests for the Audio_sensor module."""

import re
import subprocess
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pamiq_vrchat.sensors.audio import AudioSensor, get_vrchat_audio_output_device_index


class TestAudioSensor:
    """Tests for the AudioSensor class."""

    @pytest.fixture
    def mock_soundcard_audio_input(self, mocker):
        """Create a mock for SoundcardAudioInput."""
        mock = mocker.patch("pamiq_vrchat.sensors.audio.SoundcardAudioInput")
        mock_instance = mock.return_value
        # Mock read method to return a simple audio
        mock_instance.read.return_value = np.zeros((2, 512), dtype=np.uint8)
        return mock

    @pytest.fixture
    def mock_get_vrchat_audio_output_device_index(self, mocker):
        """Mock the get_vrchat_audio_output_device_index function."""
        return mocker.patch(
            "pamiq_vrchat.sensors.audio.get_vrchat_audio_output_device_index",
            return_value=2,
        )

    def test_init_with_camera_index(self, mock_soundcard_audio_input):
        """Test initialization with explicit camera index."""
        audio_output_device_index = 1
        AudioSensor(camera_index=audio_output_device_index)

        # Verify SoundcardAudioInput was called with the correct camera index
        mock_soundcard_audio_input.assert_called_once_with(audio_output_device_index)

    def test_init_without_camera_index(
        self, mock_soundcard_audio_input, mock_get_vrchat_audio_output_device_index
    ):
        """Test initialization without camera index (should use OBS virtual
        camera)."""
        AudioSensor()

        # Verify get_vrchat_audio_output_device_index was called
        mock_get_vrchat_audio_output_device_index.assert_called_once()

        # Verify SoundcardAudioInput was called with the index from get_vrchat_audio_output_device_index
        mock_soundcard_audio_input.assert_called_once_with(2)

    def test_read(self, mock_soundcard_audio_input):
        """Test the read method returns a frame."""
        sensor = AudioSensor(camera_index=0)
        frame = sensor.read()

        # Verify read was called
        mock_soundcard_audio_input.return_value.read.assert_called_once()

        # Verify frame has the expected shape and type
        assert isinstance(frame, np.ndarray)
        assert frame.shape == (2, 512)
        assert frame.dtype == np.uint8


# class TestGetObsVirtualCameraIndex:
#     """Tests for the get_vrchat_audio_output_device_index function."""

#     @pytest.fixture
#     def mock_shutil_which(self, mocker):
#         """Mock shutil.which function."""
#         return mocker.patch("pamiq_vrchat.sensors.audio.shutil.which")

#     @pytest.fixture
#     def mock_subprocess_run(self, mocker):
#         """Mock subprocess.run function."""
#         mock = mocker.patch("pamiq_vrchat.sensors.audio.subprocess.run")
#         mock_result = MagicMock()
#         mock.return_value = mock_result
#         return mock, mock_result

#     def test_v4l2_ctl_not_found(self, mock_shutil_which):
#         """Test error when v4l2-ctl is not found."""
#         mock_shutil_which.return_value = None

#         with pytest.raises(RuntimeError, match="v4l2-ctl not found"):
#             get_vrchat_audio_output_device_index()

#     def test_obs_virtual_camera_not_found(self, mock_shutil_which, mock_subprocess_run):
#         """Test error when OBS virtual camera is not found."""
#         mock_shutil_which.return_value = "/usr/bin/v4l2-ctl"  # v4l2-ctl exists
#         mock_run, mock_result = mock_subprocess_run

#         # Set up mock output without OBS Virtual Camera
#         mock_result.stdout = "Some Device\n/dev/video0\n\nAnother Device\n/dev/video1\n"

#         with pytest.raises(
#             RuntimeError, match="Can not find OBS virtual camera device"
#         ):
#             get_vrchat_audio_output_device_index()

#     def test_obs_virtual_camera_found(self, mock_shutil_which, mock_subprocess_run):
#         """Test successful finding of OBS virtual camera."""
#         mock_shutil_which.return_value = "/usr/bin/v4l2-ctl"  # v4l2-ctl exists
#         mock_run, mock_result = mock_subprocess_run

#         # Set up mock output with OBS Virtual Camera
#         mock_result.stdout = (
#             "Some Device\n/dev/video0\n\nOBS Virtual Camera\n/dev/video2\n"
#         )

#         index = get_vrchat_audio_output_device_index()

#         # Verify subprocess.run was called with the correct arguments
#         mock_run.assert_called_once_with(
#             ["v4l2-ctl", "--list-devices"],
#             capture_output=True,
#             text=True,
#             check=True,
#         )

#         # Verify the correct index was extracted
#         assert index == 2
