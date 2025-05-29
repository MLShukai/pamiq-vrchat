import logging
import sys
import threading
from typing import override

import numpy as np
from pamiq_core.interaction.modular_env import Sensor

logger = logging.getLogger(__name__)

type AudioFrame = np.typing.NDArray[np.float32]


class AudioSensor(Sensor[AudioFrame]):
    """Capturing audio from an input device with background reading.

    This sensor captures audio frames from a specified input device in a
    background thread to ensure consistent frame availability. The
    default input device is the one VRChat.exe is using, but can be
    configured by the argument.

    The sensor must be set up using setup() before reading and torn down
    using teardown() when finished to properly manage the background
    thread.
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
        self._sample_rate = sample_rate

        if device_name is None:
            device_name = get_device_name_vrchat_is_outputting_to()
        self._input = SoundcardAudioInput(
            sample_rate, device_name, block_size, channels
        )

        self._data: AudioFrame | None = None
        self._data_cv = threading.Condition()
        self._running = True
        self._reading_thread: threading.Thread | None = None

    def _reading_loop(self) -> None:
        """Background thread loop for continuous audio reading."""

        while self._running:
            data = self._input.read(self._frame_size)
            with self._data_cv:
                self._data = data
                self._data_cv.notify_all()

    @override
    def read(self) -> AudioFrame:
        """Read a frame from the background audio buffer.

        This method retrieves audio data that was captured by the background
        thread. If no data is available, it waits for up to twice the expected
        frame duration before raising an error.

        Returns:
            A numpy array containing the audio with shape (self._frame_size, channels).

        Raises:
            RuntimeError: If no audio data becomes available within the timeout period.
        """
        with self._data_cv:
            if self._data is None:
                self._data_cv.wait(self._frame_size / self._sample_rate * 2)
            if self._data is None:
                raise RuntimeError("No audio data is available")
            out, self._data = self._data, None
            return out

    @override
    def setup(self) -> None:
        """Set up the sensor and start the background reading thread.

        This method must be called before using read() to ensure the
        background audio capture thread is running.
        """
        super().setup()
        self._running = True
        self._reading_thread = threading.Thread(target=self._reading_loop)
        self._reading_thread.start()

        logger.info("Start background reading thread.")

    @override
    def teardown(self) -> None:
        """Tear down the sensor and stop the background reading thread.

        This method should be called when finished with the sensor to
        properly clean up the background thread and resources.
        """
        super().teardown()
        self._running = False
        if self._reading_thread is not None:
            self._reading_thread.join()
            self._reading_thread = None
            logger.info("End background reading thread.")

    def __del__(self) -> None:
        """Calls teardown."""
        self.teardown()


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
    import psutil
    from pycaw.pycaw import AudioUtilities

    for session in AudioUtilities.GetAllSessions():
        try:
            if session.Process and session.Process.name() == "VRChat.exe":
                print(session.Identifier.split("|")[0], end="")
                sys.exit(0)
        except psutil.NoSuchProcess:
            pass
    """).strip()

    out = subprocess.check_output([sys.executable, "-c", script], text=True).strip()
    if out:
        return out
