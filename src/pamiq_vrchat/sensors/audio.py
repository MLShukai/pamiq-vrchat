import logging
import sys
from typing import override

import numpy as np
import numpy.typing as npt
from pamiq_core.interaction.modular_env import Sensor
from pamiq_core.interaction.wrappers import Wrapper

logger = logging.getLogger(__name__)

type AudioFrame = npt.NDArray[np.float32]


class AudioSensor(Sensor[AudioFrame]):
    """Capturing audio from a input device.

    The default input device is the one VRChat.exe is using, but can be
    configured by the argument.
    """

    def __init__(
        self,
        frame_size: int,
        sample_rate: int = 16000,
        channels: int = 2,
        device_name: str | None = None,
        block_size: int | None = None,
    ):
        """Initializes an AudioSensor instance.

        Args:
            frame_size: Number of samples the user needs.
            sample_rate: Sample rate.
            channels: Audio channels.
            device_name: Audio device name for model's input. If None, automatically tries to find the device used by VRChat.exe.
            block_size: Number of samples SoundCard reads.
        """
        from pamiq_io.audio import SoundcardAudioInput

        super().__init__()
        self._frame_size = frame_size
        if device_name is None:
            device_name = get_device_name_vrchat_is_outputting_to()
        self._input = SoundcardAudioInput(
            sample_rate, device_name, block_size, channels
        )

    @override
    def read(self) -> AudioFrame:
        """Reads a frame from the Soundcard.

        Returns:
            A numpy array containing the audio with shape (self._frame_size, channels).
        """
        return self._input.read(self._frame_size)


def get_device_name_vrchat_is_outputting_to() -> str | None:
    """Find the speaker device VRChat.exe is outputting to.

    Returns:
        The device name VRChat.exe is using.
    """
    if sys.platform == "linux":
        return get_device_name_vrchat_is_outputting_to_on_linux()
    elif sys.platform == "win32":
        return get_device_name_vrchat_is_outputting_to_on_windows()
    else:
        raise RuntimeError(f"Platform {sys.platform} is not supported.")


def get_device_name_vrchat_is_outputting_to_on_linux() -> str | None:
    """Find the speaker device VRChat.exe is outputting to.

    Linux implementation.
    """
    import re
    import shutil
    import subprocess

    if shutil.which("pactl") is None:
        logger.warning("pactl command is not found.")
        return

    pactl_output = subprocess.check_output(
        ["pactl", "list", "source-outputs"], text=True
    )

    vrchat_section = re.search(
        r'Source Output #\d+.*?application\.name = "VRChat\.exe".*?',
        pactl_output,
        re.DOTALL,
    )

    if not vrchat_section:
        return None

    source_match = re.search(r"Source: (\d+)", vrchat_section.group(0))
    if not source_match:
        return None

    source_id = source_match.group(1)

    sources_output = subprocess.check_output(
        ["pactl", "list", "sources", "short"], text=True
    )

    for line in sources_output.splitlines():
        parts = line.split()
        if parts and parts[0] == source_id:
            return parts[1]

    logger.warning("Can not find speaker device VRChat.exe is outputting to.")
    return None


def get_device_name_vrchat_is_outputting_to_on_windows() -> str | None:
    """Find the speaker device VRChat.exe is outputting to on Windows.

    This function launches a separate Python subprocess to query the Windows
    audio subsystem using pycaw. This indirect approach is necessary to avoid
    a COM threading model error that occurs when calling these Windows APIs
    directly from the main application thread:

    "OSError: [WinError -2147417850] Cannot change thread mode after it is set"
    """

    import subprocess
    from textwrap import dedent

    script = dedent("""
    import sys
    from pycaw.pycaw import AudioUtilities

    for session in AudioUtilities.GetAllSessions():
        if session.Process and session.Process.name() == "VRChat.exe":
            print(session.Identifier.split("|")[0], end="")
            sys.exit(0)
    """).strip()

    out = subprocess.check_output([sys.executable, "-c", script], text=True).strip()
    if out:
        return out


class AudioLengthCompletionWrapper(Wrapper[AudioFrame, AudioFrame]):
    """Wrapper class that can complete audio length using audio output
    previously."""

    def __init__(self, frame_size: int, reset_buffer_on_pause: bool = True) -> None:
        """Initialize.

        Args:
            frame_size: Target num of samples.
            reset_buffer_on_pause: If True, make self._buffer empty when pause.
        """
        super().__init__()
        self._frame_size = frame_size
        self._reset_buffer_on_pause = reset_buffer_on_pause
        self._buffer = None

    @override
    def wrap(self, value: AudioFrame) -> AudioFrame:
        """Reads a frame from the Soundcard and outputs audio by adjusting
        length.

        Args:
            value: Audio obtained from read() of AudioSensor (shape is [AudioSensor._frame_size, channels]).
        Returns:
            Audio adjusted it's length (shape is [min(AudioSensor._frame_size, self._frame_size), channels]).
        Raises:
            ValueError: If Input shape does not match the format as [frame_size, n_channels].
        """
        if value.ndim != 2:
            raise ValueError(
                f"Input shape {value.shape} does not match the format as [frame_size, n_channels]."
            )
        if value.shape[0] >= self._frame_size:
            # no need to complete audio length
            return value[-self._frame_size :]
        if self._buffer is None:
            self._buffer = np.zeros(
                (self._frame_size, value.shape[1]), dtype=value.dtype
            )
        # complete audio length using audio output previously.
        self._buffer = np.concatenate([self._buffer, value])[-self._frame_size :]
        return self._buffer.copy()  # prevent self._buffer from broken.

    @override
    def on_paused(self) -> None:
        """Reset buffer."""
        if self._reset_buffer_on_pause:
            self._buffer = None
