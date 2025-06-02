"""Tests for the Audio_sensor module."""

import re

import numpy as np
import psutil
import pytest
from pytest_mock import MockerFixture

try:
    from pamiq_io.audio import SoundcardAudioInput
except Exception:
    pytest.skip("Can not import SoundcardAudioInput module.", allow_module_level=True)


from pamiq_vrchat.sensors.audio import (
    AudioSensor,
    get_device_name_vrchat_is_outputting_to,
    get_device_name_vrchat_is_outputting_to_on_linux,
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
    def mock_get_device_name_vrchat_is_outputting_to(self, mocker: MockerFixture):
        """Mock the get_device_name_vrchat_is_outputting_to function."""
        return mocker.patch(
            "pamiq_vrchat.sensors.audio.get_device_name_vrchat_is_outputting_to",
            return_value="vrchat_device",
        )

    def test_init_with_audio_input_device_index(self, mock_soundcard_audio_input):
        """Test initialization with explicit audio device name."""
        AudioSensor(
            sample_rate=16000,
            frame_size=FRAME_SIZE,
            channels=2,
            device_name="default",
            block_size=None,
        )
        # Verify SoundcardAudioInput was called with the correct device name
        mock_soundcard_audio_input.assert_called_once_with(16000, "default", None, 2)

    def test_init_without_audio_input_device_index(
        self,
        mock_soundcard_audio_input,
        mock_get_device_name_vrchat_is_outputting_to,
        caplog: pytest.LogCaptureFixture,
    ):
        """Test initialization without audio device name (should use the device
        being used by VRChat.exe)."""
        AudioSensor(
            sample_rate=16000,
            frame_size=FRAME_SIZE,
            channels=2,
            device_name=None,
            block_size=None,
        )
        # Verify get_device_name_vrchat_is_outputting_to was called
        mock_get_device_name_vrchat_is_outputting_to.assert_called_once()
        # Verify SoundcardAudioInput was called with the index from get_device_name_vrchat_is_outputting_to
        mock_soundcard_audio_input.assert_called_once_with(
            16000, "vrchat_device", None, 2
        )

        assert (
            "Detected audio device vrchat outputting to is 'vrchat_device'"
            in caplog.messages
        )

    def test_read(
        self, mock_soundcard_audio_input, mock_get_device_name_vrchat_is_outputting_to
    ):
        """Test the read method returns a expected frame."""
        sensor = AudioSensor(frame_size=FRAME_SIZE)
        sensor.setup()
        frame = sensor.read()
        # Verify frame has the expected shape and type
        assert frame.shape == (FRAME_SIZE, 2)
        assert frame.dtype == np.float32
        sensor.teardown()

    def test_read_returns_zero_frame_when_unavailable_and_flag_true(
        self,
        mock_soundcard_audio_input,
        mock_get_device_name_vrchat_is_outputting_to,
        mocker: MockerFixture,
    ):
        """Test that read() returns zero frame when no data is available and
        return_zero_frame_if_unavailable is True."""
        # Use small frame_size for quick timeout
        frame_size = 10
        sample_rate = 44100
        mocker.patch("threading.Condition", mocker.MagicMock())

        sensor = AudioSensor(
            frame_size=frame_size,
            sample_rate=sample_rate,
            return_zero_frame_if_unavailable=True,
        )
        # Don't call setup() so background thread doesn't start and _data remains None

        frame = sensor.read()

        # Verify zero frame is returned
        assert frame.shape == (frame_size, 2)
        assert frame.dtype == np.float32
        assert np.all(frame == 0.0)

    def test_read_raises_error_when_unavailable_and_flag_false(
        self,
        mock_soundcard_audio_input,
        mock_get_device_name_vrchat_is_outputting_to,
        mocker: MockerFixture,
    ):
        """Test that read() raises RuntimeError when no data is available and
        return_zero_frame_if_unavailable is False."""
        # Use small frame_size for quick timeout
        frame_size = 10
        sample_rate = 44100
        mocker.patch("threading.Condition", mocker.MagicMock())

        sensor = AudioSensor(
            frame_size=frame_size,
            sample_rate=sample_rate,
            return_zero_frame_if_unavailable=False,
        )
        # Don't call setup() so background thread doesn't start and _data remains None

        with pytest.raises(RuntimeError, match="No audio data is available"):
            sensor.read()


class TestGetDeviceIdVRChatIsOutputtingTo:
    def test_practical(self):
        processes = {proc.name() for proc in psutil.process_iter(["name"])}

        if "VRChat.exe" not in processes:
            pytest.skip("VRChat process is not found in practical test.")

        assert get_device_name_vrchat_is_outputting_to()


class TestGetDeviceIdVRChatIsOutputtingToOnLinux:
    """Tests for get_device_name_vrchat_is_outputting_to function."""

    def test_work_normally(self):
        get_device_name_vrchat_is_outputting_to_on_linux()

    def test_normal_case(self, mocker: MockerFixture):
        """Test when VRChat output device is found."""
        mocker.patch("shutil.which", return_value="/usr/bin/pactl")

        mock_check_output = mocker.patch("subprocess.check_output")
        mock_check_output.side_effect = [
            PACTL_SOURCE_OUTPUTS_LIST_CONTENT,
            PACTL_SOURCES_LIST_SHORT_CONTENT,
        ]

        result = get_device_name_vrchat_is_outputting_to_on_linux()

        assert result == "alsa_output.pci-0000_00_1f.3.analog-stereo.monitor"
        assert mock_check_output.call_count == 2

    def test_pactl_not_found(self, mocker: MockerFixture, caplog):
        """Test when pactl command is not found."""
        mocker.patch("shutil.which", return_value=None)

        result = get_device_name_vrchat_is_outputting_to_on_linux()

        assert result is None
        assert "pactl command is not found" in caplog.text

    def test_vrchat_not_found(self, mocker: MockerFixture):
        """Test when VRChat process is not found."""
        mocker.patch("shutil.which", return_value="/usr/bin/pactl")

        source_outputs = (
            "Source Output #100\n"
            "    Driver: protocol-native.c\n"
            '    application.name = "Discord.exe"\n'
            "    Source: 42\n"
        )

        mock_check_output = mocker.patch(
            "subprocess.check_output", return_value=source_outputs
        )

        result = get_device_name_vrchat_is_outputting_to_on_linux()

        assert result is None
        mock_check_output.assert_called_once()


PACTL_SOURCE_OUTPUTS_LIST_CONTENT = """\
Source Output #43
        Driver: protocol-native.c
        Owner Module: 10
        Client: 277
        Source: 1
        Sample Specification: s16le 2ch 44100Hz
        Channel Map: front-left,front-right
        Format: pcm, format.sample_format = "\"s16le\""  format.rate = "44100"  format.channels = "2"  format.channel_map = "\"front-left,front-right\""
        Corked: no
        Mute: no
        Volume: front-left: 65536 / 100% / 0.00 dB,   front-right: 65536 / 100% / 0.00 dB
                balance 0.00
        Buffer Latency: 3806 usec
        Source Latency: 0 usec
        Resample method: speex-float-1
        Properties:
                application.name = "OBS"
                application.icon_name = "obs"
                media.role = "production"
                media.name = "音声出力キャプチャ (PulseAudio)"
                native-protocol.peer = "UNIX socket client"
                native-protocol.version = "35"
                application.process.id = "2703144"
                application.process.user = "gop-geson"
                application.process.host = "gop-geson-01"
                application.process.binary = "obs"
                application.language = "en_US.UTF-8"
                window.x11.display = ":0"
                application.process.machine_id = "14c61fb10f5340cb9c39948358e3bab0"
                module-stream-restore.id = "source-output-by-media-role:production"

Source Output #120
        Driver: protocol-native.c
        Owner Module: 10
        Client: 941
        Source: 1
        Sample Specification: s16le 1ch 48000Hz
        Channel Map: mono
        Format: pcm, format.sample_format = "\"s16le\""  format.rate = "48000"  format.channels = "1"  format.channel_map = "\"mono\""
        Corked: no
        Mute: no
        Volume: mono: 65536 / 100% / 0.00 dB
                balance 0.00
        Buffer Latency: 3562 usec
        Source Latency: 0 usec
        Resample method: copy
        Properties:
                media.name = "audio stream #3"
                application.name = "VRChat.exe"
                native-protocol.peer = "UNIX socket client"
                native-protocol.version = "34"
                application.process.id = "2348530"
                application.process.user = "gop-geson"
                application.process.host = "gop-geson-01"
                application.process.binary = "wine64-preloader"
                application.language = "en_US.UTF-8"
                window.x11.display = ":0"
                application.process.machine_id = "14c61fb10f5340cb9c39948358e3bab0"
                module-stream-restore.id = "source-output-by-application-name:VRChat.exe"
"""


PACTL_SOURCES_LIST_SHORT_CONTENT = (
    "1\talsa_output.pci-0000_00_1f.3.analog-stereo.monitor\n"
    "43\talsa_input.pci-0000_00_1f.3.analog-stereo\n"
)
