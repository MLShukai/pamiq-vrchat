"""Tests for the Audio_sensor module."""

import re
import subprocess

import numpy as np
import pytest
from pytest_mock import mocker

from pamiq_vrchat.sensors.audio import (
    AudioSensor,
    get_device_index_vrc_is_outputting_to,
)

FRAME_SIZE = 1024


class TestAudioSensor:
    """Tests for the AudioSensor class."""

    @pytest.fixture
    def mock_soundcard_audio_input(self, mocker):
        """Create a mock for SoundcardAudioInput."""
        mock = mocker.patch("pamiq_vrchat.sensors.audio.SoundcardAudioInput")
        mock_instance = mock.return_value
        # Mock read method to return a simple audio
        mock_instance.read.return_value = np.zeros((FRAME_SIZE, 2), dtype=np.float32)
        return mock

    @pytest.fixture
    def mock_get_device_index_vrc_is_outputting_to(self, mocker):
        """Mock the get_device_index_vrc_is_outputting_to function."""
        return mocker.patch(
            "pamiq_vrchat.sensors.audio.get_device_index_vrc_is_outputting_to",
            return_value=2,
        )

    def test_init_with_audio_input_device_index(self, mock_soundcard_audio_input):
        """Test initialization with explicit camera index."""
        audio_input_device_index = 1
        AudioSensor(
            sample_rate=16000,
            frame_size=FRAME_SIZE,
            channels=2,
            device_id=audio_input_device_index,
            block_size=None,
        )
        # Verify SoundcardAudioInput was called with the correct camera index
        mock_soundcard_audio_input.assert_called_once_with(
            16000, audio_input_device_index, None, 2
        )

    def test_init_without_audio_input_device_index(
        self, mock_soundcard_audio_input, mock_get_device_index_vrc_is_outputting_to
    ):
        """Test initialization without camera index (should use OBS virtual
        camera)."""
        AudioSensor(
            sample_rate=16000,
            frame_size=FRAME_SIZE,
            channels=2,
            device_id=None,
            block_size=None,
        )
        # Verify get_device_index_vrc_is_outputting_to was called
        mock_get_device_index_vrc_is_outputting_to.assert_called_once()
        # Verify SoundcardAudioInput was called with the index from get_device_index_vrc_is_outputting_to
        mock_soundcard_audio_input.assert_called_once_with(16000, 2, None, 2)

    def test_read(self, mock_soundcard_audio_input):
        """Test the read method returns a frame."""
        sensor = AudioSensor(frame_size=FRAME_SIZE)
        frame = sensor.read()
        # Verify read was called
        mock_soundcard_audio_input.return_value.read.assert_called_once()
        # Verify frame has the expected shape and type
        assert isinstance(frame, np.ndarray)
        assert frame.shape == (FRAME_SIZE, 2)
        assert frame.dtype == np.float32


class TestGetDeviceIndexVrcIsOutputtingTo:
    """Tests for the get_device_index_vrc_is_outputting_to."""

    def test_successful_device_found(self, mocker):
        # Mock pulsectl.Pulse
        mock_pulse = mocker.patch("pulsectl.Pulse", autospec=True)
        mock_pulse_instance = mock_pulse.return_value.__enter__.return_value
        # Mock VRChat.exe's speaker device
        mock_source_output1 = mocker.Mock()
        mock_source_output1.proplist = {"application.name": "VRChat.exe"}
        mock_source_output1.source = "5"
        # Mock other application's speaker device
        mock_source_output2 = mocker.Mock()
        mock_source_output2.proplist = {"application.name": "firefox.exe"}
        mock_source_output2.source = "2"
        # Set return values from source_output_list
        mock_pulse_instance.source_output_list.return_value = [
            mock_source_output2,
            mock_source_output1,
        ]
        # Get devices
        result = get_device_index_vrc_is_outputting_to()
        # check the results
        assert result == 5
        mock_pulse.assert_called_once_with("pamiq-vrchat")
        mock_pulse_instance.source_output_list.assert_called_once()

    def test_no_vrchat_found(self, mocker):
        # Mock pulsectl.Pulse
        mock_pulse = mocker.patch("pulsectl.Pulse", autospec=True)
        mock_pulse_instance = mock_pulse.return_value.__enter__.return_value
        # Mock other application without VRChat.exe
        mock_source_output = mocker.Mock()
        mock_source_output.proplist = {"application.name": "firefox.exe"}
        mock_source_output.source = "2"
        # Set return values from source_output_list
        mock_pulse_instance.source_output_list.return_value = [mock_source_output]
        # Check if RuntimeError raises
        with pytest.raises(
            RuntimeError,
            match="Can not find speaker device VRChat.exe is outputting to.",
        ):
            get_device_index_vrc_is_outputting_to()

    def test_multiple_vrchat_instances(self, mocker):
        # Mock pulsectl.Pulse
        mock_pulse = mocker.patch("pulsectl.Pulse", autospec=True)
        mock_pulse_instance = mock_pulse.return_value.__enter__.return_value
        # Mock multiple VRChat.exe's speaker devices
        mock_source_output1 = mocker.Mock()
        mock_source_output1.proplist = {"application.name": "VRChat.exe"}
        mock_source_output1.source = "6"
        mock_source_output2 = mocker.Mock()
        mock_source_output2.proplist = {"application.name": "VRChat.exe"}
        mock_source_output2.source = "4"
        # Set return values from source_output_list
        mock_pulse_instance.source_output_list.return_value = [
            mock_source_output1,
            mock_source_output2,
        ]
        # Get devices
        result = get_device_index_vrc_is_outputting_to()
        # Check if the device index of the first VRChat found is returned
        assert result == 6

    def test_empty_source_list(self, mocker):
        # Mock pulsectl.Pulse
        mock_pulse = mocker.patch("pulsectl.Pulse", autospec=True)
        mock_pulse_instance = mock_pulse.return_value.__enter__.return_value
        # Set return values as no devices
        mock_pulse_instance.source_output_list.return_value = []
        # Check if RuntimeError raises
        with pytest.raises(
            RuntimeError,
            match="Can not find speaker device VRChat.exe is outputting to.",
        ):
            get_device_index_vrc_is_outputting_to()
