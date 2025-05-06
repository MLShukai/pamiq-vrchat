"""Tests for the Audio_sensor module."""

import numpy as np
import pytest
from pytest_mock import MockerFixture

try:
    from pamiq_io.audio import SoundcardAudioInput
except Exception:
    pytest.skip("Can not import SoundcardAudioInput module.", allow_module_level=True)


from pamiq_vrchat.sensors.audio import (
    AudioSensor,
)

FRAME_SIZE = 1024


class TestAudioSensor:
    """Tests for the AudioSensor class."""

    @pytest.fixture
    def mock_soundcard_audio_input(self, mocker: MockerFixture):
        """Create a mock for SoundcardAudioInput."""
        mock = mocker.patch("pamiq_io.audio.SoundcardAudioInput")
        mock_instance = mock.return_value
        # Mock read method to return a simple audio
        mock_instance.read.return_value = np.zeros((FRAME_SIZE, 2), dtype=np.float32)
        return mock

    @pytest.fixture
    def mock_get_device_id_vrchat_is_outputting_to(self, mocker: MockerFixture):
        """Mock the get_device_id_vrchat_is_outputting_to function."""
        return mocker.patch(
            "pamiq_vrchat.sensors.audio.get_device_id_vrchat_is_outputting_to",
            return_value="vrchat_device",
        )

    def test_init_with_audio_input_device_index(self, mock_soundcard_audio_input):
        """Test initialization with explicit camera index."""
        AudioSensor(
            sample_rate=16000,
            frame_size=FRAME_SIZE,
            channels=2,
            device_id="default",
            block_size=None,
        )
        # Verify SoundcardAudioInput was called with the correct camera index
        mock_soundcard_audio_input.assert_called_once_with(16000, "default", None, 2)

    def test_init_without_audio_input_device_index(
        self, mock_soundcard_audio_input, mock_get_device_id_vrchat_is_outputting_to
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
        # Verify get_device_id_vrchat_is_outputting_to was called
        mock_get_device_id_vrchat_is_outputting_to.assert_called_once()
        # Verify SoundcardAudioInput was called with the index from get_device_id_vrchat_is_outputting_to
        mock_soundcard_audio_input.assert_called_once_with(
            16000, "vrchat_device", None, 2
        )

    def test_read(self, mock_soundcard_audio_input):
        """Test the read method returns a frame."""
        sensor = AudioSensor(frame_size=FRAME_SIZE)
        frame = sensor.read()
        # Verify frame has the expected shape and type
        assert frame.shape == (FRAME_SIZE, 2)
        assert frame.dtype == np.float32
